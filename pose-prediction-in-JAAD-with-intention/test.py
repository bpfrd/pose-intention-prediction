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
filename_save = "checkpoint_lstm.pkl"
l2_lambda = 0.00001

JOINTS = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),(7,9),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]

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
    #images[0].save(f"{prefix}-observation.gif", save_all=True, loop=0, duration=500, append_images=images[1:])

    #images = []
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
        #last_crossing = obs_crossing[-1].unsqueeze(0)
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

def validate(model, loss_fn, loss_fn_crossing, val_loader):

    start = time.time()

    loss_val = 0.0
    ade_val = 0.0
    fde_val = 0.0
    kde_val = 0.0
    acc_val = 0.
    count_val = 0
    model.eval()
    with torch.no_grad():

        for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, obs_c, target_c, label_c) in enumerate(val_loader):
            obs_p = obs_p.to(device=device).double() 
            obs_s = obs_s.to(device=device).double()
            target_p = target_p.to(device=device).double() 
            target_s = target_s.to(device=device).double()
            obs_c = obs_c.to(device=device).double()
            target_c = target_c.to(device=device).double()
  
            pred_s, pred_crossing = model(obs_s=obs_s, obs_crossing=obs_c)
            loss = loss_fn(pred_s, target_s) + loss_fn_crossing(pred_crossing, target_c)
            
            pred_p = obs_p[-1].unsqueeze(0) + torch.cumsum(pred_s.detach(), dim=0)

            batch = obs_p.size(1)
            count_val += batch
            loss_val += loss.item()*batch
            ade_val += float(ade(pred_p, target_p))
            fde_val += float(fde(pred_p, target_p))
            kde_val += float(kde(pred_p, target_p))
            acc_val += (torch.argmax(pred_crossing[-1], dim=-1) == torch.argmax(target_c[-1], dim=-1)).sum()

        visualization(obs_p[:,0], obs_f[0], pred_p[:,0], target_p[:,0], target_f[0], "visualization")

        loss_val /= count_val
        ade_val /= count_val
        fde_val /= count_val
        kde_val /= count_val
        acc_val /= count_val

    print("date: %s "%datetime.datetime.now(),
          "|loss_v: %0.4f "%loss_val,
          "|fde_v: %0.4f "%fde_val,
          "|ade_v: %0.4f "%ade_val, 
          "|kde_v: %0.4f "%kde_val, 
          "|acc_v: %0.4f "%acc_val,
          "|time(s): %0.2f "%(time.time()-start)) 

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

        model.train()
        for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, obs_c, target_c, label_c) in enumerate(train_loader):


            obs_p = obs_p.to(device=device).double() 
            obs_s = obs_s.to(device=device).double()
            target_p = target_p.to(device=device).double() 
            target_s = target_s.to(device=device).double()
            obs_c = obs_c.to(device=device).double()
            target_c = target_c.to(device=device).double()
   
            pred_s, pred_crossing = model(obs_s=obs_s, obs_crossing=obs_c)
            loss = loss_fn(pred_s, target_s)
            loss_c = 10*loss_fn_crossing(pred_crossing, target_c)
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()) 
            loss = loss + loss_c + l2_lambda * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_p = obs_p[-1].unsqueeze(0) + torch.cumsum(pred_s.detach(), dim=0)
            #pred_p[0] = obs_p[-1] + pred_s[0]
            #pred_p[1] = pred_p[0] + pred_s[1] = obs_p[-1] + pred_s[0] + pred_s[1] = 
            #pred_p[2] = pred_p[1] + pred_s[2] = obs_p[-1] + pred_s[0] + pred_s[1] + pred_s[2]
            #      ...
            #pred_p[i] = pred_p[i-1] + pred_s[i] = obs_p[-1] + torch.cumsum(pred_s, dim=0)
            
            batch = obs_p.size(1)
            count_train += batch 
            loss_crossing_train += loss_c.item()*batch
            loss_train += loss.item()*batch
            ade_train += float(ade(pred_p, target_p))
            fde_train += float(fde(pred_p, target_p))
            kde_train += float(kde(pred_p, target_p))
            acc_train += (torch.argmax(pred_crossing[-1], dim=-1) == torch.argmax(target_c[-1], dim=-1)).sum()

        loss_train /= count_train
        loss_crossing_train /= count_train
        ade_train /= count_train
        fde_train /= count_train    
        kde_train /= count_train    
        acc_train /= count_train
        #scheduler.step(loss_train)

        #if(epoch % 100 == 0):
        #    visualization(obs_p[:,0], obs_f[0], pred_p[:,0], target_p[:,0], target_f[0], "unconverged")

        loss_val = 0.0
        ade_val = 0.0
        fde_val = 0.0
        kde_val = 0.0
        acc_val = 0.
        count_val = 0
        model.eval()
        with torch.no_grad():

            for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, obs_c, target_c, label_c) in enumerate(val_loader):
                obs_p = obs_p.to(device=device).double() 
                obs_s = obs_s.to(device=device).double()
                target_p = target_p.to(device=device).double() 
                target_s = target_s.to(device=device).double()
                obs_c = obs_c.to(device=device).double()
                target_c = target_c.to(device=device).double()
  
                pred_s, pred_crossing = model(obs_s=obs_s, obs_crossing=obs_c)
                loss = loss_fn(pred_s, target_s) + loss_fn_crossing(pred_crossing, target_c)
                
                pred_p = obs_p[-1].unsqueeze(0) + torch.cumsum(pred_s.detach(), dim=0)

                batch = obs_p.size(1)
                count_val += batch
                loss_val += loss.item()*batch
                ade_val += float(ade(pred_p, target_p))
                fde_val += float(fde(pred_p, target_p))
                kde_val += float(kde(pred_p, target_p))
                acc_val += (torch.argmax(pred_crossing[-1], dim=-1) == torch.argmax(target_c[-1], dim=-1)).sum()

            loss_val /= count_val
            ade_val /= count_val
            fde_val /= count_val
            kde_val /= count_val
            acc_val /= count_val

        if(loss_val < loss_val_best):
            loss_val_best = loss_val
            print(f'Saving ... at epoch {epoch}')
            torch.save(model.state_dict(), filename_save)

        if(epoch % 10 == 0):
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
                  "|g_norm: %0.8f"%np.sqrt(sum([p.grad.data.norm(2).item() for p in model.parameters()])),
                  "|l_reg: %0.8f"%l2_norm,
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
model = LSTM(pose_dim=34, h_dim=64, num_layers=1, dropout=0.).to(device).double()

#loss_fn = nn.L1Loss() 
loss_fn = nn.MSELoss() 
loss_fn_crossing = nn.BCELoss()

model.load_state_dict(torch.load(filename_save))

#loading the data
args.dtype = 'val'
args.save_path = args.save_path.replace('train', 'val')
args.file = args.file.replace('train', 'val')
val_loader = DataLoader.data_loader(args)

print('validating ...')
validate(model, loss_fn, loss_fn_crossing, val_loader)

