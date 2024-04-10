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
import copy
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter 

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
    for num in range(img.size(0)):
        for i in range(3):
            array[num][i] = array[num][i]*std[i]
        for i in range(3):
            array[num][i] = array[num][i]+mean[i]
    #cv2.add(data_array, mean, data_array)
    #print(data_array)
    array = (array/255.0)
    data = torch.from_numpy(array).float()
    return data
    
def denormilize(img):
    array = np.asarray(img.detach().cpu())
    array = (array*255.0)
    for num in range(img.size(0)):
        for i in range(3):
            array[num][i] = array[num][i]-mean[i]
        for i in range(3):
            array[num][i] = array[num][i]/std[i]
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
          4,
          2,
          dist=False)
          #seed=cfg.get('seed', None))
          #shuffle=False)
     return engine, dataloader

def up_prepare(config):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    scale_factor = config.scale_factor
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()

	# DCE_net.apply(weights_init)
    if config.load_pretrain == True:
         DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    return DCE_net
     
def up_loss(enhanced_image, lowlight_image, A):
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    # L_exp = Myloss.L_exp(16,0.6)
    L_TV = Myloss.L_TV()
    E = 0.6
    Loss_TV = 1600*L_TV(A)
    loss_spa = torch.mean(L_spa(enhanced_image, lowlight_image))
    loss_col = 5*torch.mean(L_color(enhanced_image))

    loss_exp = 10*torch.mean(L_exp(enhanced_image,E))
    return Loss_TV + loss_spa + loss_col + loss_exp

def train(config):

    DCE_net = up_prepare(config)

    writer = SummaryWriter("/home/msai/jwang098/logs/")


    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
    DCE_net.train()
    

    conf = Config.fromfile(config.down_config)
    engine, dloader = down_prepare(conf, config.down_checkpoint)

    for epoch in range(config.num_epochs):
        for iteration, data in enumerate(dloader):
            #torch.autograd.set_detect_anomaly(True)
            #if iteration == 1:
                #print(data)
            en_data = copy.deepcopy(data)
            data_low = en_data['img'].data[0]
            data_lowlight = renormilize(data_low)
            data_lowlight = data_lowlight.cuda()
            #if iteration == 1:
                #print(data_lowlight)
            enhanced_image,A  = DCE_net(data_lowlight)
            en_data['img'].data[0] = denormilize(enhanced_image)
            if iteration == 10:
                torchvision.utils.save_image(enhanced_image, "/home/msai/jwang098/tinaface/Zero-DCE++/result/epoch%d.jpg"%epoch)
            #with torch.no_grad():
            loss_down = engine(en_data)['loss']
            #loss_down = 5*torch.pow(down/2-0.4,2)
            #loss_down = down
            
            #loss_up = up_loss(enhanced_image, data_lowlight, A)
            L_color = Myloss.L_color()
            L_spa = Myloss.L_spa()
            L_exp = Myloss.L_exp(16)
            # L_exp = Myloss.L_exp(16,0.6)
            L_TV = Myloss.L_TV()
            E = 0.6
            Loss_TV = 1600*L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, data_lowlight))
            loss_col = 5*torch.mean(L_color(enhanced_image))
        
            loss_exp = 10*torch.mean(L_exp(enhanced_image,E))


            # best_loss
            loss_up = Loss_TV + loss_spa + loss_col + loss_exp
            loss =  2*loss_up + loss_down
            
            optimizer.zero_grad()
            
            #with torch.autograd.detect_anomaly():
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
            optimizer.step()
            
            writer.add_scalar('loss', loss.item(), epoch*len(dloader)+iteration)
            writer.add_scalar('Loss_TV', Loss_TV.item(), epoch*len(dloader)+iteration)
            writer.add_scalar('loss_spa', loss_spa.item(), epoch*len(dloader)+iteration)
            writer.add_scalar('loss_col', loss_col.item(), epoch*len(dloader)+iteration)
            writer.add_scalar('loss_exp', loss_exp.item(), epoch*len(dloader)+iteration)
            writer.add_scalar('loss_up', loss_up.item(), epoch*len(dloader)+iteration)
            writer.add_scalar('loss_down', loss_down.item(), epoch*len(dloader)+iteration)
            #writer.add_scalar('down', down.item(), epoch*len(dloader)+iteration)
            
            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
                print("Loss_down at iteration", iteration+1, ":", loss_down.item())
            #if iteration == 1:
                #print(data)
                #print("-------------------------------")
            if ((iteration+1) % config.snapshot_iter) == 0:
                #writer.add_scalar('loss', loss.item(), epoch*len(dloader)+iteration)
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    #parser.add_argument('--train_batch_size', type=int, default=8)
    #parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    #parser.add_argument('--snapshot_iter', type=int, default=500)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots_Zero_DCE++/stan.pth")
    parser.add_argument('--down_config', type=str, default= "/home/msai/jwang098/tinaface/configs/trainval/tinaface/tinaface_dcn.py")
    parser.add_argument('--down_checkpoint', type=str, default= "/home/msai/jwang098/tinaface/workdir/tinaface_r50_fpn_bn/tinaface_r50_fpn_gn_dcn.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)








	
