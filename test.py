import numpy as np
import os
import time
import torch
import logging
from model.PLholonet import PLholonet
from utils.dataset import create_dataloader_qis
from utils.utilis import PCC,PSNR,accuracy,random_init,tensor2value,plotcube
from torch.optim import Adam
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import re
if __name__ == "__main__":
    # load the test dataset
    batch_sz = 1
    kt=30
    ks=2
    val_data_path = "/home/zhangyp/PycharmProjects/PLholo/syn_data/data/val_Nz32_Nxy128_kt30_ks2_ppv2e-04~1e-03"
    # val_data_path = "/Users/zhangyunping/PycharmProjects/PLholo/syn_data/data/Nz25_Dz0.75e-3_ppv2e-4/val_Nz25_Nxy128_kt30_ks2"
    # val_data_path = "/home/zhangyp/PycharmProjects/PLholo/syn_data/data/train_Nz25_Nxy128_kt30_ks2"
    data_loader,dataset = create_dataloader_qis(val_data_path,batch_sz,kt,ks)
    K1_map, label, otf3d, y = next(iter(data_loader))

    #load the model
    # model_path = "/home/zhangyp/PycharmProjects/PLholo/experiment/PLholonet_Nz25_Nxy128_L5_B18_lr0.0001_G0.001_kt30_ks2"
    # model_path = "/Users/zhangyunping/PycharmProjects/PLholo/experiment/PLholonet_Nz25_Nxy128_L5_B18_lr0.0001_G1e-05_kt30_ks2"
    model_path = "/experiment/PLholonet_Nz32_Nxy128_L5_B16_lr0.0001_G0.0001_kt30_ks2_MSE"
    model_name = model_path.split('/')[-1]
    params = model_name.split('_')
    try:
        Nd = [eval(re.findall(r'L(\d+)',x)[0]) for x in params if re.findall(r'L(\d+)',x)][0]
        Nz = [eval(re.findall(r'Nz(\d+)',x)[0]) for x in params if re.findall(r'Nz(\d+)',x)][0]
        gamma = [eval(re.findall(r'G(\d+)',x)[0]) for x in params if re.findall(r'G(\d+)',x)][0]
    except:
        Nd = 5
        Nz = 25
        gamma = 0.0001
        print("cannot retrieve the params from the model name, use the default as Nd={:d} Nz={:d}".format(Nd,Nz))

    model = PLholonet(n=Nd, d=Nz, sysloss_param=gamma)
    last_path = os.path.join(model_path, 'last.pt')
    best_path = os.path.join(model_path, 'best.pt')

    try:
        state_dict = torch.load(best_path)
        model.device = torch.device('cuda')
    except:
        state_dict = torch.load(best_path,map_location='cpu')
        model.device = torch.device('cpu')
    model.load_state_dict(state_dict['param'])

    # evaluation and visualization the results
    model.eval()
    from train_verbose import visual_after_epoch
    visual_after_epoch(model,data_loader,1,model_path)
    # with torch.no_grad():
    #     K1_map = K1_map.to(torch.float32).to(device=model.device)
    #     otf3d = otf3d.to(torch.complex64).to(device=model.device)
    #     label = label.to(torch.float32).to(device=model.device)
    #     x, _sloss = model(K1_map, otf3d)
    #     _dloss = torch.mean(torch.pow(x - label, 2))
    #     _total_loss = _dloss + _sloss
    #
    #     _pcc = PCC(x, label)
    #     _psnr = PSNR(x, label)
    #     _acc = accuracy(x, label)
    #
    #     print(('\n' + '%10s' * 7) % ('  ', 'sloss', 'dloss', 'loss', 'acc', 'pcc', 'psnr'))
    #     info = ('%10s' + '%10.4g' * 6) % ('Test_result',_sloss, _dloss,
    #                                       _total_loss, _acc , _pcc, _psnr)
    #     print(info)
    #     if len(x.shape)==4:
    #         x = x.squeeze(0)
    #     pred_cube = torch.zeros_like(x)
    #     idx = (x>=0.5)
    #     pred_cube[idx] =  torch.ones_like(x)[idx]
    #     pred_cube = tensor2value(pred_cube)
    #     plotcube(pred_cube,'predicted')
    #
    #
    #     gt = tensor2value(label.squeeze(0))
    #     plotcube(gt,'gt')





