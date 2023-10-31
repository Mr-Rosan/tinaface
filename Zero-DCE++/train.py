import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import cv2
import numpy as np
from torchvision import transforms

#from torch.utils.tensorboard import SummaryWriter 

from vedacore.misc import Config, load_weights, ProgressBar, mkdir_or_exist
from vedacore.fileio import dump
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine
from vedacore.parallel import MMDataParallel


mean = [123.675, 116.28, 103.53]
std = [1, 1, 1]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def renormilize(img):
    array = np.asarray(img)
    for i in range(3):
        array[0][i] = array[0][i]*std[i]
    for i in range(3):
        array[0][i] = array[0][i]+mean[i]
    #cv2.add(data_array, mean, data_array)
    #print(data_array)
    array = (array/255.0)
    data = torch.from_numpy(array).float()
    return data

def down_prepare(cfg, checkpoint):
     engine = build_engine(cfg.down_engine)
     load_weights(engine.model, checkpoint, map_location='cpu')
     device = torch.cuda.current_device()
     engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])
     dataset = build_dataset(cfg.data.train)
     dataloader = build_dataloader(
          dataset,
          1,
          1,
          dist=False,
          shuffle=False)
     return engine, dataloader

def train(config):

    os.environ['CUDA_VISIBLE_DEVICES']='0'
    scale_factor = config.scale_factor
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()

	# DCE_net.apply(weights_init)
    if config.load_pretrain == True:
         DCE_net.load_state_dict(torch.load(config.pretrain_dir))

    #writer = SummaryWriter("/home/msai/jwang098/logs/")

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    # L_exp = Myloss.L_exp(16,0.6)
    L_TV = Myloss.L_TV()
    L_down = Myloss.L_down()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
    DCE_net.train()
    

    conf = Config.fromfile(config.down_config)
    engine, dloader = down_prepare(conf, config.down_checkpoint)

    for epoch in range(config.num_epochs):
        for iteration, data in enumerate(dloader):
            #with torch.no_grad():
            loss_down = engine(data)['loss']
            data_low = data['img'].data[0]
            data_lowlight = renormilize(data_low)
            data_lowlight = data_lowlight.cuda()
            enhanced_image,A  = DCE_net(data_lowlight)
            E = 0.6
            Loss_TV = 1600*L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, data_lowlight))
            loss_col = 5*torch.mean(L_color(enhanced_image))
    
            loss_exp = 10*torch.mean(L_exp(enhanced_image,E))
    
            # best_loss
            loss =  Loss_TV + loss_spa + loss_col + loss_exp + loss_down
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
            optimizer.step()
    
            #writer.add_scalar('loss', loss.item(), iteration)
            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
            if ((iteration+1) % config.display_iter) == 0:
                print("Loss_down at iteration", iteration+1, ":", loss_down.item())
            if ((iteration+1) % config.snapshot_iter) == 0:
    	        torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 	




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= True)
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots_Zero_DCE++/Epoch99.pth")
    parser.add_argument('--down_config', type=str, default= "/home/msai/jwang098/tinaface/configs/trainval/tinaface/tinaface_dcn.py")
    parser.add_argument('--down_checkpoint', type=str, default= "/home/msai/jwang098/tinaface/workdir/tinaface_r50_fpn_bn/tinaface_r50_fpn_gn_dcn.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)








	