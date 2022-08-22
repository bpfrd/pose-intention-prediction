import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import DataLoader
import datetime
from PIL import Image, ImageDraw

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
filename_save = "decouple-lstm-vae.pkl"
l2_lambda = 0.00001

JOINTS = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),(7,9),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]

def decouple(x, left_hip=11, right_hip=12):
    seq_len, batch, l = x.shape
    x_local = x.view(seq_len, batch, l//2, 2)
    x_global = 0.5*(x_local[:, :, left_hip] + x_local[:, :, right_hip])
    x_local = x_local - x_global.unsqueeze(2)
    return x_global, x_local.view(seq_len, batch, l)

def recouple(x_global=None, x_local=None, left_hip=11, right_hip=12):
    seq_len, batch, l = x_local.shape
    return (x_local.view(seq_len, batch, l//2, 2) + x_global.unsqueeze(2) ).view(seq_len, batch, l)

def visualization(obs_p, obs_scenes, pred_p, true_p, scenes, prefix):
    obs_len, l = obs_p.shape
    obs = obs_p.view(obs_len, l//2, 2)
    pred_len, l = pred_p.shape
    pred = pred_p.view(pred_len, l//2, 2)
    true = true_p.view(pred_len, l//2, 2)

    images = []
    for time in range(obs_len):
        im = Image.open(obs_scenes[time].replace("JAAD/scene/", "video_"))
        img = ImageDraw.Draw(im)
        for joint in JOINTS:
            obs_points = (obs[time, joint[0],0], obs[time, joint[0], 1], obs[time, joint[1],0], obs[time, joint[1], 1])
            img.line(obs_points, fill="blue", width=5)
        images.append(im)

    for time in range(pred_len):
        im = Image.open(scenes[time].replace("JAAD/scene/", "video_"))
        img = ImageDraw.Draw(im)
        for joint in JOINTS:
            pred_points = (pred[time, joint[0],0], pred[time, joint[0], 1], pred[time, joint[1],0], pred[time, joint[1], 1])
            true_points = (true[time, joint[0],0], true[time, joint[0], 1], true[time, joint[1],0], true[time, joint[1], 1])
            img.line(pred_points, fill="red", width=5)
            img.line(true_points, fill="green", width=5)
        images.append(im)
    #images[0].save(f"{prefix}-future.gif", save_all=True, loop=0, duration=500, append_images=images[1:])
    images[0].save(f"{prefix}.gif", save_all=True, loop=0, duration=500, append_images=images[1:])

def kde(pred, true):
    #kde defined in Mangalem's paper
    seq_len, batch, l = true.shape
    return torch.sum(torch.sum( torch.sum(torch.abs(pred-true), dim=-1) / l//2, dim=0)/seq_len)

def ade(pred, true):
    seq_len, batch, l = true.shape
    return torch.sum(torch.sqrt(torch.sum((pred-true)**2, dim=-1) / l//2 ))/seq_len

def fde(pred, true):
    seq_len, batch, l = true.shape
    return torch.sum(torch.sqrt(torch.sum((pred[-1]-true[-1])**2, dim=-1) / l//2 ))


class Encoder(nn.Module):
    def __init__(self, pose_dim=34, h_dim=32, latent_dim=16, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()

        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC_mean = nn.Linear(h_dim, latent_dim)
        self.FC_var = nn.Linear(h_dim, latent_dim)

    def forward(self, obs_s=None):
        batch = obs_s.size(1)
        state_tuple = (torch.zeros(self.num_layers, batch, self.h_dim, device=self.device, dtype=torch.float64),
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.device, dtype=torch.float64))
        output, state_tuple = self.encoder(obs_s, state_tuple)
        return self.FC_mean(output[-1]), self.FC_var(output[-1]) #mean, log_var

class Decoder(nn.Module):
    def __init__(self, pose_dim=34, embedding_dim=8, h_dim=32, latent_dim=16, num_layers=1, dropout=0.3):
        super(Decoder, self).__init__()
        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.decoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC = nn.Sequential(nn.Linear(latent_dim, h_dim))
        self.mlp = nn.Sequential(nn.Linear(h_dim, pose_dim))
        
    def forward(self, obs_s=None, latent=None, pred_len=None):

        batch = obs_s.size(1)
        state_tuple = (self.FC(latent).unsqueeze(0).repeat(self.num_layers, 1, 1), 
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.device, dtype=torch.float64))

        last_s = obs_s[-1].unsqueeze(0)
        preds_s = torch.tensor([], device=self.device, dtype=torch.float64)
        for _ in range(pred_len):
            output, state_tuple = self.decoder(last_s, state_tuple)
            curr_s = self.mlp(output.view(-1, self.h_dim))
            preds_s = torch.cat((preds_s, curr_s.unsqueeze(0)), dim=0)
            last_s = curr_s.unsqueeze(0)
       
        return preds_s

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, obs_s=None, pred_len=15):
        mean, log_var = self.Encoder(obs_s=obs_s)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        preds_s = self.Decoder(obs_s=obs_s, latent=z, pred_len=pred_len)

        return preds_s, mean, log_var

def vae_loss_function(x, x_hat, mean, log_var):
    # BCE_loss = nn.BCELoss()
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    assert x_hat.shape == x.shape
    reconstruction_loss = torch.mean(torch.norm(x - x_hat, dim=len(x.shape) - 1))
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + 0.01*KLD 


class LSTM(nn.Module):
    def __init__(self, pose_dim=34, h_dim=16, num_layers=1, dropout=0.2):
        super(LSTM, self).__init__()

        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.device = device 

        #pose forecasting network arch
        self.encoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.decoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(h_dim, pose_dim))

        #crossing forecasting network arch
        self.decoder_crossing = nn.LSTM(pose_dim, h_dim, num_layers=1, dropout=0.) 
        self.fc_crossing = nn.Sequential(nn.Linear(h_dim, 2), nn.Softmax(dim=-1))
        self.mlp_crossing = nn.Linear(h_dim, h_dim)
        self.embedding = nn.Linear(2, pose_dim)

    def forward(self, obs_s=None, obs_crossing=None, pred_len=15):

        seq_len, batch, l = obs_s.shape
        assert(l == self.pose_dim)
        state_tuple = (torch.zeros(self.num_layers, batch, self.h_dim, device=self.device, dtype=torch.float64),
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.device, dtype=torch.float64))

        output, state_tuple = self.encoder(obs_s, state_tuple)

        pred_s = torch.tensor([], device=self.device, dtype=torch.float64)
        last_s = obs_s[-1].unsqueeze(0)

        pred_crossing = torch.tensor([], device=self.device, dtype=torch.float64)
        last_crossing = obs_s[-1].unsqueeze(0)
        #last_crossing = self.embedding(obs_crossing[-1]).unsqueeze(0)
        #last_crossing = self.embedding(torch.zeros(1, batch, 2, device=self.device, dtype=torch.float64))
        state_tuple_crossing = (self.mlp_crossing(state_tuple[0]), 
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.device, dtype=torch.float64))

        for i in range(pred_len):
             
             output_crossing, state_tuple_crossing = self.decoder_crossing(last_crossing, state_tuple_crossing)
             curr_crossing = self.fc_crossing(output_crossing[-1])
             pred_crossing = torch.cat((pred_crossing, curr_crossing.unsqueeze(0)), dim=0)
             #last_crossing = curr_crossing.detach().unsqueeze(0)
             last_crossing = self.embedding(curr_crossing).unsqueeze(0)

             output, state_tuple = self.decoder(last_s, state_tuple)
             curr_s = self.fc(output[-1])
             pred_s = torch.cat((pred_s, curr_s.unsqueeze(0)), dim=0)
             #last_s = curr_s.detach().unsqueeze(0)
             last_s = curr_s.unsqueeze(0)

        return pred_s, pred_crossing


def training_loop(n_epochs, optimizer, scheduler, model, loss_fn, loss_fn_crossing, train_loader, val_loader):

    global l2_lambda 

    loss_val_best = 1e100
    for epoch in range(1, n_epochs + 1):

        loss_train = 0.0
        loss_crossing_train = 0.0
        ade_train = 0.0
        fde_train = 0.0
        kde_train = 0.0
        acc_train = 0.
        acc_state_train = 0.
        count_train = 0
        start = time.time()

        if epoch == 1:
            for g in optimizer.param_groups:
                g['lr'] = 0.01
        elif epoch == 5:
            for g in optimizer.param_groups:
                g['lr'] = 0.001
            l2_lambda *= 1.5
        elif epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] = 0.0001
            l2_lambda *= 2

        [net.train() for (key, net) in model.items() if 'net' in key]
        for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, obs_c, target_c, label_c) in enumerate(train_loader):


            obs_p = obs_p.to(device=device).double() 
            obs_s = obs_s.to(device=device).double()
            target_p = target_p.to(device=device).double() 
            target_s = target_s.to(device=device).double()
            obs_c = obs_c.to(device=device).double()
            target_c = target_c.to(device=device).double()

            obs_s_global, obs_s_local = decouple(obs_s)
            target_s_global, target_s_local = decouple(target_s)

            pred_s_global, pred_crossing = model['net_global'](obs_s=obs_s_global, obs_crossing=obs_c)
            loss_global = loss_fn(pred_s_global, target_s_global)
            loss_c = 100*loss_fn_crossing(pred_crossing, target_c)

            pred_s_local, mean, log_var = model['net_local'](obs_s=obs_s_local)
            loss_local = vae_loss_function(target_s_local, pred_s_local, mean, log_var) 

            l2_norm = 0. #sum(p.pow(2.0).sum() for p in [net.parameters() for net in model.values()]) 
            loss = loss_global + loss_local + loss_c + l2_lambda * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_s = recouple(pred_s_global, pred_s_local)
            pred_p = obs_p[-1].unsqueeze(0) + torch.cumsum(pred_s.detach(), dim=0)
            
            batch = obs_p.size(1)
            count_train += batch 
            loss_crossing_train += loss_c.item()*batch
            loss_train += loss.item()*batch
            ade_train += float(ade(pred_p, target_p))
            fde_train += float(fde(pred_p, target_p))
            kde_train += float(kde(pred_p, target_p))
            acc_train += (torch.argmax(pred_crossing[-1], dim=-1) == torch.argmax(target_c[-1], dim=-1)).sum()
            acc_state_train += (torch.argmax(pred_crossing, dim=-1).view(-1) == torch.argmax(target_c, dim=-1).view(-1)).sum()

        loss_train /= count_train
        loss_crossing_train /= count_train
        ade_train /= count_train
        fde_train /= count_train    
        kde_train /= count_train    
        acc_train /= count_train
        acc_state_train /= (count_train*target_c.size(1))
        #scheduler.step(loss_train)

        #if(epoch % 100 == 0):
        #    visualization(obs_p[:,0], obs_f[0], pred_p[:,0], target_p[:,0], target_f[0], "visualization")

        loss_val = 0.0
        ade_val = 0.0
        fde_val = 0.0
        kde_val = 0.0
        acc_val = 0.
        acc_state_val = 0.
        count_val = 0
        [net.eval() for (key, net) in model.items() if 'net' in key]
        with torch.no_grad():

            for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, obs_c, target_c, label_c) in enumerate(val_loader):
                obs_p = obs_p.to(device=device).double() 
                obs_s = obs_s.to(device=device).double()
                target_p = target_p.to(device=device).double() 
                target_s = target_s.to(device=device).double()
                obs_c = obs_c.to(device=device).double()
                target_c = target_c.to(device=device).double()

                obs_s_global, obs_s_local = decouple(obs_s)
                target_s_global, target_s_local = decouple(target_s)

                pred_s_global, pred_crossing = model['net_global'](obs_s=obs_s_global, obs_crossing=obs_c)
                pred_s_local, mean, log_var = model['net_local'](obs_s=obs_s_local)
                pred_s = recouple(pred_s_global, pred_s_local)
                loss = loss_fn(pred_s_global, target_s_global) + \
                               loss_fn(pred_s_local, target_s_local) +\
                                     loss_fn_crossing(pred_crossing, target_c)
                
                pred_p = obs_p[-1].unsqueeze(0) + torch.cumsum(pred_s.detach(), dim=0)

                batch = obs_p.size(1)
                count_val += batch
                loss_val += loss.item()*batch
                ade_val += float(ade(pred_p, target_p))
                fde_val += float(fde(pred_p, target_p))
                kde_val += float(kde(pred_p, target_p))
                acc_val += (torch.argmax(pred_crossing[-1], dim=-1) == torch.argmax(target_c[-1], dim=-1)).sum()
                acc_state_val += (torch.argmax(pred_crossing, dim=-1).view(-1) == torch.argmax(target_c, dim=-1).view(-1)).sum()

            loss_val /= count_val
            ade_val /= count_val
            fde_val /= count_val
            kde_val /= count_val
            acc_val /= count_val
            acc_state_val /= (count_val*target_c.size(1))

        if(loss_val < loss_val_best):
            loss_val_best = loss_val
            print(f'Saving ... at epoch {epoch}')
            #torch.save(model['params'].state_dict(), filename_save)
            torch.save({
                'local_state_dict': model['net_local'].state_dict(),
                'global_state_dict': model['net_global'].state_dict(),
                }, filename_save)

        if(epoch % 10 == 0):
            #print(f'Saving ... at epoch {epoch}')
            #torch.save(model.state_dict(), filename_save)
            print("date: %s "%datetime.datetime.now(),
                  "|e: %d "%epoch,
                  "|loss_t: %0.4f "%loss_train,
                  "|loss_v: %0.4f "%loss_val,
                  "|best_loss_v: %0.4f "%loss_val_best,
                  "|fde_t: %0.4f "%fde_train,
                  "|fde_v: %0.4f "%fde_val,
                  "|ade_t: %0.4f "%ade_train, 
                  "|ade_v: %0.4f "%ade_val, 
                  "|kde_t: %0.4f "%kde_train, 
                  "|kde_v: %0.4f "%kde_val, 
                  "|acc_t: %0.4f "%acc_train,
                  "|acc_v: %0.4f "%acc_val,
                  "|acc_state_t: %0.4f "%acc_state_train,
                  "|acc_state_v: %0.4f "%acc_state_val,
                  "|loss_crossing_t: %0.4f "%loss_crossing_train,
                  "|time(s): %0.2f "%(time.time()-start)) 


class Args():
   def __init__(self, batch_size=200, loader_shuffle=True, pin_memory=False, loader_workers=1, load_checkpoint=False,
             jaad_dataset='',
             dtype='train', from_file=True, save=False, 
             file='jaad_train_I15_O15_S15_shuffled.csv',
             save_path='',
             input=15, output=15, stride=15, skip=1, task='bounding_box-intention-body_pos', use_scenes=False):

       self.batch_size = batch_size
       self.loader_shuffle = loader_shuffle
       self.pin_memory = pin_memory
       self.loader_workers = loader_workers
       self.load_checkpoint = load_checkpoint
       self.jaad_dataset = jaad_dataset
       self.dtype = dtype
       self.from_file = from_file
       self.save = save
       self.file = file
       self.save_path = save_path
       self.input = input
       self.output = output
       self.stride = stride
       self.skip = skip
       self.task = task
       self.use_scenes = use_scenes

args = Args()

#defining the model
#vae for local 
encoder = Encoder(pose_dim=34, h_dim=32, latent_dim=16, num_layers=1, dropout=0.)
decoder = Decoder(pose_dim=34, embedding_dim=8, h_dim=32, latent_dim=16, num_layers=1, dropout=0.)
vae = VAE(encoder, decoder).to(device).double()
#lstm for global & intention
lstm = LSTM(pose_dim=2, h_dim=32, num_layers=1, dropout=0.).to(device).double()

model_params = list(lstm.parameters()) + list(vae.parameters())
model = {'net_global': lstm, 'net_local': vae, 'params': model_params}

#loss_fn = nn.L1Loss() 
loss_fn = nn.MSELoss() 
loss_fn_crossing = nn.BCELoss()

optimizer = optim.Adam(model_params, lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, min_lr=1e-5, verbose=True)

if(args.load_checkpoint):
    #model.load_state_dict(torch.load(filename_save))
    checkpoint = torch.load(filename_save)
    model['net_global'].load_state_dict(checkpoint['global_state_dict'])
    model['net_local'].load_state_dict(checkpoint['local_state_dict'])

#loading the data
args.dtype = "train"
train_loader = DataLoader.data_loader(args)
args.dtype = 'val'
args.save_path = args.save_path.replace('train', 'val')
args.file = args.file.replace('train', 'val')
val_loader = DataLoader.data_loader(args)


print('Training ...')
training_loop(n_epochs=1000000,
           optimizer=optimizer, 
           scheduler=scheduler,
           model=model,
           loss_fn=loss_fn,
           loss_fn_crossing=loss_fn_crossing,
           train_loader=train_loader,
           val_loader=val_loader)

