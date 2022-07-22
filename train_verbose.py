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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, opt, dataloader, epoch, freeze = []):
    model.train()

    for k,v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    nbatch = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nbatch)
    logger.info('\n Training========================================')
    logger.info(('\n'+'%10s'*8)%('Epoch  ','GPU_memory','c_sloss','c_dloss','loss','acc','pcc','psnr'))
    total_loss = []
    sloss = []
    dloss = []
    acc =[]
    pcc = []
    psnr = []
    opt.zero_grad()
    for i,(K1_map, label, otf3d, _) in pbar:
        K1_map = K1_map.to(torch.float32).to(device=model.device)
        otf3d = otf3d.to(torch.complex64).to(device=model.device)
        label = label.to(torch.float32).to(device=model.device)
        x, _sloss = model(K1_map,otf3d)
        # _dloss = torch.mean(torch.pow(x-label,2))
        _dloss = torch.nn.BCELoss()(x,label)
        _total_loss = _dloss+_sloss
        _total_loss.backward()
        opt.step()

        # metric calculation
        _pcc = PCC(x,label)
        _psnr = PSNR(x,label)
        _acc = accuracy(x,label)

        # update metric
        total_loss.append(tensor2value(_total_loss))
        sloss.append(tensor2value(_sloss))
        dloss.append(tensor2value(_dloss))
        pcc.append(tensor2value(_pcc))
        psnr.append(tensor2value(_psnr))
        acc.append(tensor2value(_acc))

        # printing
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

        info = ('%10s'*2 + '%10.4g'*6)%('%g'%(epoch),mem,_sloss,_dloss,np.mean(total_loss),_acc,_pcc,_psnr)
        pbar.set_description(info)


    return np.mean(sloss),np.mean(dloss),np.mean(total_loss),np.mean(acc),np.mean(pcc),np.mean(psnr)

def eval_epoch(model, opt, dataloader, epoch):
    model.eval()
    nbatch = len(dataloader)
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nbatch)
    # logger.info('\n Evaluation========================================')
    # logger.info(('\n'+'%10s'*7)%('Epoch  ','GPU_memory','c_sloss','c_dloss','loss','pcc','psnr'))
    total_loss = []
    sloss = []
    dloss = []
    pcc = []
    psnr = []
    acc=[]

    with torch.no_grad():
        for i,(K1_map, label, otf3d, _) in pbar:
            K1_map = K1_map.to(torch.float32).to(device=model.device)
            otf3d = otf3d.to(torch.complex64).to(device=model.device)
            label = label.to(torch.float32).to(device=model.device)
            x, _sloss = model(K1_map,otf3d)
            # _dloss = torch.mean(torch.pow(x-label,2))
            _dloss = torch.nn.BCELoss()(x, label)
            _total_loss = _dloss+_sloss

            # metric
            _pcc = PCC(x,label)
            _psnr = PSNR(x,label)
            _acc = accuracy(x,label)

            # printing
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

            # info = ('%10s'*2 + '%10.4g'*5)%('%g'%(epoch),mem,stage_symlosses,loss_discrepancy,avg_loss,pcc_,psnr_)
            # pbar.set_description(info)
            pbar.set_description("Evaluating.....")
            total_loss.append(tensor2value(_total_loss))
            sloss.append(tensor2value(_sloss))
            dloss.append(tensor2value(_dloss))
            pcc.append(tensor2value(_pcc))
            psnr.append(tensor2value(_psnr))
            acc.append(tensor2value(_acc))


    logger.info(('\n'+'%10s'*7)%('  ','sloss','dloss','loss','acc','pcc','psnr'))
    info = ('%10s'+ '%10.4g'*6)%('Eval_result',np.mean(sloss),np.mean(dloss),
                                 np.mean(total_loss),np.mean(acc), np.mean(pcc),np.mean(psnr))
    logger.info(info)

    return np.mean(sloss),np.mean(dloss),np.mean(total_loss),np.mean(acc),np.mean(pcc),np.mean(psnr)

def visual_after_epoch(model,dataloader,epoch,out_dir=None):
    K1_map, label, otf3d, y = next(iter(dataloader))
    # evaluation and visualization the results
    model.eval()
    with torch.no_grad():
        K1_map = K1_map.to(torch.float32).to(device=model.device)
        otf3d = otf3d.to(torch.complex64).to(device=model.device)
        label = label.to(torch.float32).to(device=model.device)
        x, _sloss = model(K1_map, otf3d)
        if len(x.shape)==4:
            x = x[0,:,:,:]
            label = label[0,:,:,:]
        pred_cube = torch.zeros_like(x)
        idx = (x>=0.5)
        filename = os.path.join(out_dir,"Epoch{:d}".format(epoch))
        pred_cube[idx] =  torch.ones_like(x)[idx]
        pred_cube = tensor2value(pred_cube)
        plotcube(pred_cube,'predicted',filename+"_pred.png",show=False)
        gt = tensor2value(label)
        plotcube(gt,'gt',filename+"_gt.png",show=False)


if __name__=="__main__":
    random_init(seed=43)

    parser = ArgumentParser(description='PLholonet')
    parser.add_argument('--batch_sz', type=int, default=16, help='batch size')
    parser.add_argument('--train_data_path',type=str,default='train_Nz32_Nxy128_kt30_ks2_ppv2e-04~1e-03',help='datapath with params')
    parser.add_argument('--val_data_path',type=str,default='val_Nz32_Nxy128_kt30_ks2_ppv2e-04~1e-03',help='datapath with params')
    parser.add_argument('--data_root',type=str,default='./syn_data/data',help='data root')
    # parser.add_argument('--obj_type', type=str, default='sim', help='exp or sim')
    parser.add_argument('--Nz', type=int, default=25, help='depth number')
    parser.add_argument('--kt', type=int, default=30, help='temporal oversampling ratio')
    parser.add_argument('--ks', type=int, default=2, help='spatial oversampling ratio')
    parser.add_argument('--dz', type=str, default='1200um', help='depth interval')
    parser.add_argument('--ppv', type=str, default='5e-03', help='ppv')
    parser.add_argument('--lr_init', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='epochs')
    parser.add_argument('--Nxy', type=int, default=128, help='lateral size')
    parser.add_argument('--gamma', type=float, default=1e-4, help='symmetric loss parameter')
    parser.add_argument('--layer_num', type=int, default=5,  help='phase number of PLholoNet')
    parser.add_argument("--visualization",action='store_true',default=True,help='whether output visualization results during training')
    # args = parser.parse_args([])
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])


    Nd = args.layer_num
    batch_sz = args.batch_sz
    lr = args.lr_init
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    gamma = args.gamma


    try:
        print("Compile the params from the dataset")
        data_name = train_data_path.split('/')[-1]
        params = data_name.split('_')
        Nz = [eval(re.findall(r'Nz(\d+)',x)[0]) for x in params if re.findall(r'Nz(\d+)',x)][0]
        Nxy = [eval(re.findall(r'Nxy(\d+)',x)[0]) for x in params if re.findall(r'Nxy(\d+)',x)][0]
        kt = [eval(re.findall(r'kt(\d+)',x)[0]) for x in params if re.findall(r'kt(\d+)',x)][0]
        ks = [eval(re.findall(r'ks(\d+)',x)[0]) for x in params if re.findall(r'ks(\d+)',x)][0]

    except:
        print("Loading the default value:")
        Nz = args.Nz
        kt = args.kt
        ks = args.ks
        Nxy = args.Nxy

    sys_param = 'Nz' + str(Nz)  + '_Nxy' + str(Nxy) + \
                '_L' + str(Nd) + '_B' + str(batch_sz) + \
                '_lr' + str(lr) + '_G' + str(gamma) + '_kt' + str(kt)+'_ks' + str(ks)
    train_data_path = os.path.join(args.data_root,train_data_path)
    val_data_path = os.path.join(args.data_root,val_data_path)

    out_dir = './experiment/'
    log_dir = './logs/'
    model_name =  'PLholonet_'+ sys_param

    save_dir = out_dir + model_name
    log_file = log_dir+model_name +'.log'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)


    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(message)s",level=logging.INFO)
    formater = logging.Formatter("%(message)s")
    # define the filehaddler for writing log in file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formater)
    # define the StreamHandler for writing on the screen
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formater)
    # add both handlers
    logger.addHandler(fh)
    # logger.addHandler(sh)



    logger.info("The args are following:")
    logger.info(args)

    tb_writer = SummaryWriter(save_dir)
    last_path = os.path.join(save_dir, 'last.pt')
    best_path = os.path.join(save_dir, 'best.pt')


    #%% Dataset prepare
    train_dataloader, train_dataset = create_dataloader_qis(train_data_path,batch_sz,kt,ks)
    val_dataloader, val_dataset = create_dataloader_qis(val_data_path,batch_sz,kt,ks)

    model = PLholonet(n=Nd, d=Nz, sysloss_param=args.gamma)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.module.to("cuda")
        model.device = torch.device('cuda')
    else:
        model = torch.nn.DataParallel(model)
        model.device = torch.device('cpu')


    optimizer = Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience =3, verbose=True)
    # otf3d = obj3d(wave_length = 633*nm, img_rows = 64, img_cols=64, slice=30,size = 10*mm, depth = 2*cm).get_otf3d()

    # resume
    start_epoch = 0
    end_epoch = start_epoch+args.epochs
    scheduler.last_epoch = start_epoch - 1
    max_acc = 0

    for epoch in range(start_epoch, end_epoch, 1):
        train_out = train_epoch(model,optimizer,train_dataloader,epoch)
        eval_out = eval_epoch(model,optimizer,val_dataloader,epoch)

        # Log
        current_lr = optimizer.param_groups[0]['lr']
        tags = ['train/sloss', 'train/dloss', 'train/total_loss',  'train/acc', 'train/pcc','train/psnr'# train loss & metric
                                                                                            'val/sloss', 'val/dloss', 'val/total_loss',  'val/acc','val/pcc','val/psnr' # val loss & metric
                ]  # params
        for x, tag in zip(list(train_out[:-1]) + list(eval_out[:-1]) , tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard

        # save the last ckpt
        ckpt = {
            'param':model.state_dict(),
            'model_name': model_name,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt,last_path)
        logger.info("\n Epoch {:d} saved".format(epoch))

        # update the best
        if eval_out[3] > max_acc:
            max_acc = eval_out[3]
        if max_acc == eval_out[3]:
            torch.save(ckpt,best_path)
            logger.info("Best updated at Epoch {:d}".format(epoch))

        if args.visualization and epoch%10==0:
            visual_after_epoch(model,val_dataloader,epoch,save_dir)

