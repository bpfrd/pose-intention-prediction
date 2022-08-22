import torch
import torchvision
import torchvision.transforms.functional as TF
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import time

class myJAAD(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')
        
        if(args.from_file):
            sequence_centric = pd.read_csv(args.file) #.sample(10)
            df = sequence_centric.copy()      
            for v in list(df.columns.values):
                print(v+' loaded')
                try:
                    df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
                except:
                    continue
            sequence_centric[df.columns] = df[df.columns]
            
        else:
            print('Not available')
            exit()
 
        self.data = sequence_centric.copy().reset_index(drop=True)
        self.args = args
        self.dtype = args.dtype
        print(args.dtype, "set loaded")
        print('*'*30)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        seq = self.data.iloc[index]
        outputs = []
        ########################################################## 
        scene_paths = [os.path.join("/work/vita/datasets/JAAD/JAAD_clips_bodyposes_by_openpifpaf/images/"+seq["scenefolderpath"][frame],"0"+ seq["filename"][frame]) 
                           for frame in range(0,self.args.input,self.args.skip)]
        scene_paths_t = [os.path.join("/work/vita/datasets/JAAD/JAAD_clips_bodyposes_by_openpifpaf/images/"+seq["scenefolderpath_t"][frame], "0"+seq["filename_t"][frame]) 
                           for frame in range(0,self.args.output,self.args.skip)]
      
        observed_keypoints = torch.tensor(np.array(seq.keypoints))
        future_keypoints = torch.tensor(np.array(seq.future_keypoints))
        obs_keypoints = torch.tensor([seq.keypoints[i] for i in range(0,self.args.input,self.args.skip)])
        obs_speed_keypoints = (obs_keypoints[1:] - obs_keypoints[:-1])
        outputs.append(obs_speed_keypoints.type(torch.float32))
        if 'body_pos' in self.args.task:
            true = torch.tensor([seq.future_keypoints[i] for i in range(0,self.args.output,self.args.skip)])
            true_speed = torch.cat(((true[0]-obs_keypoints[-1]).unsqueeze(0), true[1:]-true[:-1]))
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs_keypoints.type(torch.float32))
            outputs.append(true.type(torch.float32))

        ########################################################## 

        observed = torch.tensor(np.array(seq.bounding_box))
        future = torch.tensor(np.array(seq.future_bounding_box))
        
        obs = torch.tensor([seq.bounding_box[i] for i in range(0,self.args.input,self.args.skip)])
        obs_speed = (obs[1:] - obs[:-1])
    
        outputs.append(obs_speed.type(torch.float32))
        
        if 'bounding_box' in self.args.task:
            true = torch.tensor([seq.future_bounding_box[i] for i in range(0,self.args.output,self.args.skip)])
            true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1])) #Question
            outputs.append(true_speed.type(torch.float32))
            outputs.append(obs.type(torch.float32))
            outputs.append(true.type(torch.float32))
        
        if 'intention' in self.args.task:
            obs_cross = torch.tensor([seq.crossing_obs[i] for i in range(0,self.args.input,self.args.skip)])
            obs_non_cross = torch.ones(obs_cross.shape, dtype=torch.int64)-obs_cross
            obs_cross = torch.cat((obs_non_cross.unsqueeze(1), obs_cross.unsqueeze(1)), dim=1)

            true_cross = torch.tensor([seq.crossing_true[i] for i in range(0,self.args.output,self.args.skip)])
            true_non_cross = torch.ones(true_cross.shape, dtype=torch.int64)-true_cross
            true_cross = torch.cat((true_non_cross.unsqueeze(1), true_cross.unsqueeze(1)), dim=1)
            cross_label = torch.tensor(seq.label)
            outputs.append(obs_cross.type(torch.float32))
            outputs.append(true_cross.type(torch.float32))
            outputs.append(cross_label.type(torch.float32))

        if self.args.use_scenes:     
            scene_paths = [os.path.join(seq["scenefolderpath"][frame], '%.4d'%seq.ID, seq["filename"][frame]) 
                           for frame in range(0,self.args.input,self.args.skip)]
        
            for i in range(len(scene_paths)):
                scene_paths[i] = scene_paths[i].replace('haziq-data', 'smailait-data').replace('scene', 'resized_scenes')

            scenes = torch.tensor([])
            for i, path in enumerate(scene_paths):
                scene = Image.open(path)
                #bb = obs[i,:]
                #img = ImageDraw.Draw(scene)   
                #utils.drawrect(img, ((bb[0]-bb[2]/2, bb[1]-bb[3]/2), (bb[0]+bb[2]/2, bb[1]+bb[3]/2)), width=5)
                scene = self.scene_transforms(scene)
                scenes = torch.cat((scenes, scene.unsqueeze(0)))
                
            outputs.insert(0, scenes)
        outputs.insert(0, scene_paths)
        outputs.insert(1, scene_paths_t)
        
        return tuple(outputs)

    def scene_transforms(self, scene):  
        #scene = TF.resize(scene, size=(self.args.image_resize[0], self.args.image_resize[1]))
        scene = TF.to_tensor(scene)
        
        return scene

def my_collate(batch):

    (obs_f, target_f, obs_s, target_s, obs_p, target_p, _, _, _, _, obs_c, target_c, label_c) = zip(*batch)
    obs_p = torch.stack(obs_p).permute(1,0,2)
    obs_s = torch.stack(obs_s).permute(1,0,2)
    target_p = torch.stack(target_p).permute(1,0,2)
    target_s = torch.stack(target_s).permute(1,0,2)
    obs_c = torch.stack(obs_c).permute(1,0,2)
    target_c = torch.stack(target_c).permute(1,0,2)
    #label_c = torch.stack(label_c)
    out = [obs_p, obs_s, obs_f, target_p, target_s, target_f, obs_c, target_c, label_c]
    return tuple(out)

    
def data_loader(args):
    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, collate_fn=my_collate, drop_last=False)

    return dataloader
