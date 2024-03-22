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
import dataset
from PGNet import PGNet
from edge_prediction import label_edge_prediction 
from tqdm import *

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int) 
    parser.add_argument('--batchsize', default=-1, type=int)
    parser.add_argument('--savepath', default="../model/baseline", type=str)  
    parser.add_argument('--datapath', default="../data/DUTS-TR", type=str) 
    parser.add_argument('--model', default="model-32", type=str) 
    parser.parse_args()
    return parser.parse_args()

def test(Dataset, Network):
    # dataset
    args = parser()
    print(torch.cuda.device_count())

    cfg = Dataset.Config(datapath=args.datapath,mode='test')
    data = Dataset.Data(cfg)
    loader = torch.utils.data.DataLoader(data,
                                         batch_size = 1,
                                         shuffle=False,
                                         num_workers=2
                                         )

    model = args.model
    TestDataset = args.datapath.split('/')[-1]
    net = Network(cfg)
    net.load_state_dict(torch.load('../model/'+model)['net']) 
    net.train(False)
    net = net.cuda()
    test_count = 0
    for step, (image, mask, shape, name) in tqdm(enumerate(loader)):
        image, mask = image.float().cuda(), mask.float().cuda() 
        edge = label_edge_prediction(mask)  

        p1,outs = net(image)

        out_resize   = F.interpolate(p1,size=shape, mode='bilinear')
        pred   = torch.sigmoid(out_resize[0,0])
        pred  = (pred*255).cpu().detach().numpy()
        test_count += 1
        save_path = '../result/'+TestDataset+'/'+model
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+'/'+name[0]+'.png', np.round(pred))


if __name__ == '__main__':
    test(dataset, PGNet)
