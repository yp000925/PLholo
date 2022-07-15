import numpy as np
import os
import time
import torch
import logging
from model.PLholonet import PLholonet
from utils.dataset import create_dataloader_qis
from utils.utilis import PCC,PSNR,accuracy,random_init,tensor2value
from torch.optim import Adam
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

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
        _dloss = torch.mean(torch.pow(x-label,2))
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
            _dloss = torch.mean(torch.pow(x-label,2))
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

if __name__=="__main__":
    random_init(seed=43)

    parser = ArgumentParser(description='PLholonet')
    parser.add_argument('--batch_sz', type=int, default=18, help='batch size')
    # parser.add_argument('--obj_type', type=str, default='sim', help='exp or sim')
    parser.add_argument('--Nz', type=int, default=25, help='depth number')
    parser.add_argument('--kt', type=int, default=30, help='temporal oversampling ratio')
    parser.add_argument('--ks', type=int, default=2, help='spatial oversampling ratio')
    parser.add_argument('--dz', type=str, default='1200um', help='depth interval')
    parser.add_argument('--ppv', type=str, default='5e-03', help='ppv')
    parser.add_argument('--lr_init', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=250, help='epochs')
    parser.add_argument('--Nxy', type=int, default=128, help='lateral size')
    parser.add_argument('--gamma', type=float, default=1e-3, help='symmetric loss parameter')
    parser.add_argument('--layer_num', type=int, default=5,  help='phase number of PLholoNet')
    args = parser.parse_args([])
    batch_sz = args.batch_sz
    kt = args.kt
    ks = args.ks
    lr = args.lr_init
    Nz = args.Nz
    Nd = args.layer_num
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(message)s",level=logging.INFO)

    sys_param = 'Nz' + str(args.Nz)  + '_Nxy' + str(args.Nxy) + \
                '_L' + str(args.layer_num) + '_B' + str(args.batch_sz) + \
                '_lr' + str(args.lr_init) + '_G' + str(args.gamma) + '_kt' + str(args.kt)+'_ks' + str(args.ks)

    train_data_path =  './syn_data/data/train_' + 'Nz' + str(args.Nz) + '_Nxy' + str(args.Nxy)+'_kt' + str(args.kt)+'_ks' + str(args.ks)
    val_data_path = './syn_data/data/val_' + 'Nz' + str(args.Nz) + '_Nxy' + str(args.Nxy)+'_kt' + str(args.kt)+'_ks' + str(args.ks)

    out_dir = './experiment/'
    log_dir = './logs/'
    model_name =  'PLholonet_'+ sys_param

    save_dir = out_dir + model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

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


