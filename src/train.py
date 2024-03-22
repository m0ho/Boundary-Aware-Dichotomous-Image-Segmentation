#!/usr/bin/python3
# coding=utf-8
import cv2
import numpy as np
import sys
import datetime
import argparse
import os
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from BADIS import BADIS
from apex import amp
import torch.distributed as dist
from utils.lr_scheduler import LR_Scheduler
from edge_prediction import label_edge_prediction
def flat(mask):
    batch_size = mask.shape[0]
    h = 28
    mask = F.interpolate(mask,size=(int(h),int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1) 
    g = x @ x.transpose(-2,-1)
    g = g.unsqueeze(1)
    return g

def att_loss(pred,mask,p4,p5):
    g = flat(mask)
    np4 = torch.sigmoid(p4.detach())
    np5 = torch.sigmoid(p5.detach())
    p4 = flat(np4)
    p5 = flat(np5)
    w1  = torch.abs(g-p4)
    w2  = torch.abs(g-p5)
    w = (w1+w2)*0.5+1
    attbce=F.binary_cross_entropy_with_logits(pred, g,weight =w*1.0,reduction='mean')
    return attbce
    
def bce_iou_loss(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask,size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int) 
    parser.add_argument('--batchsize', default=-1, type=int)
    parser.add_argument('--savepath', default="../model/baseline", type=str)  
    parser.add_argument('--datapath', default="../data/DUTS-TR", type=str) 
    parser.parse_args()
    return parser.parse_args()

def train(Dataset, Network):
    # dataset
    args = parser()
    print(torch.cuda.device_count())
    # ############################################################
    cfg = Dataset.Config(datapath=args.datapath, savepath=args.savepath,mode='train', batch=args.batchsize, lr=0.015, momen=0.9,
                         decay=5e-4, epoch=64)#, snapshot=args.checkpoint)
    data = Dataset.Data(cfg)
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=args.batchsize,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=data.collate,
                                         )


    net = Network(cfg)

    
    base, head = [], []
    for name, param in net.named_parameters():
        if 'swin' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    scheduler = LR_Scheduler('cos',cfg.lr,cfg.epoch,len(loader),warmup_epochs=cfg.epoch//2)
    ##############################
    start_epoch = 0
    # 用于接着训练
    # start_epoch = 56
    # checkpoint  = torch.load('../model/model-'+str(start_epoch)+'.pth')
    # net.load_state_dict(checkpoint['net']) 
    # optimizer.load_state_dict(checkpoint['optimizer']) 
    ##############################
    global_step = 0
    net.train(True)
    net = net.cuda()
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')



    for epoch in range(start_epoch,cfg.epoch):
        net.train()
        test_count = 0
        for step, (image, mask) in enumerate(loader):

            image, mask = image.float().cuda(), mask.float().cuda() 
            edge = label_edge_prediction(mask)  
            optimizer.zero_grad()      
            scheduler(optimizer,step,epoch)
            p1,outs = net(image)

            loss1s = bce_iou_loss(outs[0], mask)
            loss2s = bce_iou_loss(outs[1], mask)
            loss3s = bce_iou_loss(outs[2], mask)

            loss1e = bce_iou_loss(outs[3], edge)
            loss2e = bce_iou_loss(outs[4], edge)
            loss3e = bce_iou_loss(outs[5], edge)
            
            bce_iou_all = loss1s + loss1e + 0.6*(loss2s + loss2e) + 0.4*(loss3s + loss3e)

            loss1 = bce_iou_all
            loss = loss1

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()     
            print(f'%s | step:%d/%d/%d | lr=%.16f  loss=%.6f ' % (datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[1]['lr'],loss.item()),flush = True)
            global_step += 1

        if epoch >= 50 :
                checkpoint = {
                                "net": net.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                "epoch": epoch
                            }
                if not os.path.exists(cfg.savepath):
                    os.makedirs(cfg.savepath)
                torch.save(checkpoint, cfg.savepath + '/model-' + str(epoch + 1)+'.pth')


if __name__ == '__main__':
    train(dataset, BADIS)
