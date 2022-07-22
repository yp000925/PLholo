"""
settting:
- Python 3.8.13
- torch 1.11.0
- cuda 11.3
- cudnn 8.2

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utilis import FT2d,iFT2d,autopad
from utils.dataset import create_dataloader_qis

class PLholonet_block(nn.Module):
    def __init__(self,d, alpha=4):
        super(PLholonet_block, self).__init__()
        self.K = alpha # = Ks^2 when generate the signal, K/alpha = 1
        self.alpha = alpha
        self.rho1 = torch.nn.Parameter(torch.randn([1]),requires_grad=True)
        torch.nn.init.normal_(self.rho1)
        self.rho2 = torch.nn.Parameter(torch.randn([1]),requires_grad=True)
        torch.nn.init.normal_(self.rho2)
        # self.lamda = torch.nn.Parameter(torch.randn([1]),requires_grad=True).to(device)
        # torch.nn.init.normal_(self.lamda)
        #self.denoiser = ResUNet().to(device)
        self.denoiser = denoiser(d)

    def batch_forward_proj(self,field_batch, otf3d_batch, intensity=True,scale = True):
        '''

        :param field_batch:  3d field for batch [B, C, H, W]
        :param otf3d_batch: OTF3d for batch [B, C, H, W] or [C, H, W]
        :return: holo_batch [B,1, H, W]
        '''
        [B, C, H, W] = field_batch.shape
        if len(otf3d_batch.shape) == 3:
            otf3d_batch = otf3d_batch.unsqueeze(0)
        assert otf3d_batch.shape[1] == field_batch.shape[1],"The depth slice does not match between field and OTF"
        Fholo3d =  torch.mul(FT2d(field_batch),otf3d_batch)
        Fholo = torch.sum(Fholo3d,dim = 1,keepdim=True)
        holo = iFT2d(Fholo)
        if intensity:
            holo = torch.abs(holo)
            if scale:
                # max_range = C
                # holo = holo/max_range
                # assert holo.max()[0]< 1.0,"The inner hologram is larger than 1"
                mintmp = holo.view([B,1,H*W]).min(2,keepdim=True)[0].unsqueeze(-1)
                maxtmp = holo.view([B,1,H*W]).max(2,keepdim=True)[0].unsqueeze(-1)
                holo = (holo-mintmp)/(maxtmp-mintmp)
                return holo
            else:
                return holo
        return holo

    def batch_back_proj(self, holo_batch,otf3d_batch,real_constraint= True):
        """

        :param holo_batch: holo_batch [B,1, H, W] or [B,H,W]
        :param otf3d_batch: OTF3d for batch [B, C, H, W] or [C, H, W]
        :return:
        """
        if len(holo_batch.shape) == 3:
            holo_batch = holo_batch.unsqueeze(1) #[B,1,H,W]
        if len(otf3d_batch.shape) == 3:
            otf3d_batch = otf3d_batch.unsqueeze(0)
        holo_batch = holo_batch.to(torch.complex64)
        conj_otf3d = torch.conj(otf3d_batch)
        volumne_slice = otf3d_batch.shape[1]
        holo_expand = holo_batch.tile([1,volumne_slice,1,1])
        holo_expand = FT2d(holo_expand)
        field_ft = torch.multiply(holo_expand,conj_otf3d)
        field3d = iFT2d(field_ft)
        if real_constraint:
            return torch.real(field3d)
        return field3d



    # def forward_prop(self, field_batch,otf3d):
    #     '''
    #
    #     :param field: 3d field for batch [B, C, H, W]
    #     :param otf3d: H(kx,ky,kz) [B,C, H, W]
    #     :return: holo_batch [B,1, H, W]
    #     '''
    #     if len(otf3d.shape) == 3:
    #         otf3d = otf3d.unsqueeze(0)
    #     if otf3d.shape[1] != field_batch.shape[1]:
    #         print("The depth slice does not match")
    #         raise ValueError
    #     # batch_size = field_batch.shape[0]
    #     # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
    #     holo = torch.real(torch.sum(torch.mul(otf3d, field_batch), dim=1, keepdim=True))
    #     return holo
    #
    # def back_prop(self,holo,otf3d):
    #     '''
    #
    #     :param holo: hologram
    #     :param otf3d: H(kx,ky,kz)
    #     :return: Ht*holo
    #     '''
    #
    #     if len(otf3d.shape) == 3:
    #         otf3d = otf3d.unsqueeze(0)
    #     if len(holo.shape) == 3:
    #         holo = holo.unsqueeze(1) #[B,1,H,W]
    #     holo = holo.to(torch.complex64)
    #     conj_otf3d = torch.conj(otf3d)
    #     volumne_slice = otf3d.shape[1]
    #     # perform iFT(FT(o)*conj(h))
    #     holo_expand = holo.tile([1,volumne_slice,1,1])
    #     holo_expand = FT2d(holo_expand)
    #     field_ft = torch.multiply(holo_expand,conj_otf3d)
    #     field3d = iFT2d(field_ft)
    #     return torch.real(field3d)

    def X_update(self, phi, z, u1, u2, otf3d):
        "proximal operator for forward propagation "
        x1 = phi-u1
        x2 = z-u2
        #numerator n = F( alpha * At * I_h + v)
        temp = self.batch_back_proj(x1, otf3d)
        n = self.rho1*temp+self.rho2*x2
        n = FT2d(n.to(torch.complex64))

        #denominator d = (|OTF|^2 + 1)
        otf_square = torch.abs(otf3d)**2
        ones_array = torch.ones_like(otf_square)
        d =  ones_array*self.rho2+otf_square*self.rho1
        d = d.to(torch.complex64)

        #final fraction
        x_next = iFT2d(n/d)
        return x_next.real

    def Phi_update(self,x,z,u1,u2,otf3d, K1):
        """
        proximal operator for truncated Poisson signal
        :param x: [B,D,H,W]
        :param z:
        :param u1: [B,1,H,W]
        :param u2: []
        :param otf3d:[B,D,H,W]
        :param K1: [B,1,H,W]
        :return: phi_next [B,1,H,W]
        """
        # batch_size = x.shape[0]
        # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
        phi_tilde = self.batch_forward_proj(x,otf3d)+u1
        # phi_tilde = torch.abs(phi_tilde)
        K0 = 1 - K1 # number of zero in each pixel

        ind_0 = (K1==0)# the index of pixel that does not need to update the predicted bit value (0 or already goes to optimized solution)
        ind_1 = (K1!=0) # the index of pixel that needs to update the predicted bit value
        func = lambda y: self.alpha/self.K*(K0-K1/(torch.exp(self.alpha/self.K*y)-1))+self.rho1*(y-phi_tilde)

        # when K1 = 0 the solution is directly calculated as
        phi_next = torch.ones_like(phi_tilde)
        phi_next[ind_0] = phi_tilde[ind_0]-K0[ind_0]/self.rho1

        # when K1 !=0 solve the one-dimensional equation
        phimin = 1e-5*torch.ones_like(phi_tilde, dtype=phi_next.dtype)
        phimax = 100*torch.ones_like(phi_tilde, dtype=phi_next.dtype)
        phiave = (phimin + phimax) / 2.0
        for i in range(30):
            tmp = func(phiave)
            ind_pos = torch.logical_and(tmp>0,ind_1)
            ind_neg = torch.logical_and(tmp<0,ind_1)
            ind_zero = torch.logical_and(tmp==0,ind_1)
            ind_0 = torch.logical_or(ind_0,ind_zero)
            ind_1 = torch.logical_not(ind_0)

            phimin[ind_pos] = phiave[ind_pos]
            phimax[ind_neg] = phiave[ind_neg]
            phiave[ind_1] = (phimin[ind_1]+phimax[ind_1])/2.0

        phi_next[K1 != 0] = phiave[K1 != 0]
        return phi_next


    def Z_update(self,x,phi,u1,u2,otf3d):
        [B,C,W,H] = x.shape
        z_tilde = x+u2
        # z_tilde = z_tilde.view([B*C,1,W,H])
        z_next,stage_symloss = self.denoiser(z_tilde)
        return z_next,stage_symloss

    def forward(self,x,phi,z,u1,u2,otf3d, K1):
        # t0 = time.time()
        # U, Z and X updates
        x = self.X_update(phi, z, u1, u2, otf3d)
        # t1 = time.time()
        # print(t1-t0,x.shape,phi.shape,z.shape,u1.shape,u2.shape)
        phi = self.Phi_update(x, z, u1, u2, otf3d, K1)
        # t2 = time.time()
        # print(t2-t1,x.shape,phi.shape,z.shape,u1.shape,u2.shape)
        z, stage_symloss = self.Z_update(x,phi,u1,u2,otf3d)
        # t3 = time.time()
        # print(t3-t2,x.shape,phi.shape,z.shape,u1.shape,u2.shape)
        # Lagrangian updates
        # batch_size = x.shape[0]
        # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
        u1 = u1 + phi - self.batch_forward_proj(x,otf3d)
        u2 = u2 + z - x
        # print(stage_symloss.shape)
        return x,phi,z,u1,u2,stage_symloss


class PLholonet(nn.Module):
    def __init__(self,n,d,sysloss_param = 2e-3):
        super(PLholonet, self).__init__()
        self.n = n
        self.blocks = nn.ModuleList([])
        for i in range(n):
            self.blocks.append(PLholonet_block(d))
        self.Batchlayer = torch.nn.BatchNorm2d(d)
        self.Activation = torch.nn.Sigmoid()
        self.sysloss_param = sysloss_param


    def forward(self, K1, otf3d):
        """
        :param K1: Number of ones in each pixel shape [B,1,W/K,H/K]
        :return:
        """
        # initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = K1.device
        x = self.blocks[0].batch_back_proj(K1,otf3d)
        phi = Variable(K1.data.clone()).to(device)
        z = Variable(x.data.clone()).to(device)
        u1 =torch.zeros(K1.size()).to(device)
        u2 = torch.zeros(x.size()).to(device)
        stage_symlosses = torch.tensor([0.0]).to(device)

        for i in range(self.n):
            x,phi,z,u1,u2,stage_symloss = self.blocks[i](x,phi,z,u1,u2,otf3d,K1) #x,phi,z,u1,u2,otf3d, K1
            stage_symlosses += torch.mean(torch.pow(stage_symloss,2))
            # print('stage',i,'\n symloss',stage_symlosses)

        x = self.Batchlayer(x)
        x = self.Activation(x)
        return x, self.sysloss_param*stage_symlosses/self.n

class resblock(nn.Module):
    def __init__(self,c):
        super(resblock, self).__init__()
        self.CBL1 = CBL(c, c,k=3,s=1,activation=True)
        self.CBL2 = CBL(c, c,k=3,s=1,activation=False)
        self.act = nn.LeakyReLU()

    def forward(self,x):
        return self.act(x+self.CBL2(self.CBL1(x)))


class CBL(nn.Module):
    def __init__(self,c1,c2,k=1,s=1,padding=None,g=1,activation=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1,c2,k,s,autopad(k,padding),bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if activation is True else(activation if isinstance(activation,nn.Module) else nn.Identity())
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class SoftThreshold(nn.Module):
    def __init__(self):
        super(SoftThreshold, self).__init__()

        self.soft_thr = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def forward(self, x):
        return torch.mul(torch.sign(x),torch.nn.functional.relu(torch.abs(x)-self.soft_thr))

class denoiser(nn.Module):
    def __init__(self,c):
        super(denoiser, self).__init__()
        self.resblock1 = resblock(c)
        self.resblock2 = resblock(c)
        self.soft_thr = SoftThreshold()

    def forward(self,xin):
        x = self.resblock1(xin)
        x_thr = self.soft_thr(x)
        x_out = self.resblock2(x_thr)
        x_forward_backward = self.resblock2(x)
        stage_symloss = x_forward_backward-xin
        return x_out,stage_symloss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # K1 = torch.randn([4,1,256,256])
    # otf3d = obj3d(wave_length = 633*nm, img_rows = 256, img_cols=256, slice=5,size = 10*mm, depth = 2*cm).get_otf3d()
    # net = PLholonet(n=2,d=5).to(device)
    # # x,phi,z,u1,u2 = net(K1,otf3d)
    # x, stage_symlosses = net(K1,otf3d)
    path = "/Users/zhangyunping/PycharmProjects/PLholo/syn_data/data/Nz25_Dz0.75e-3_ppv2e-4/train_Nz25_Nxy128_kt30_ks2"
    dataloader, dataset = create_dataloader_qis(path,batch_size=2,Kt=30,Ks=2)
    model = PLholonet(n=5, d=25)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.module.to("cuda")
        model.device = torch.device('cuda')
    else:
        model = torch.nn.DataParallel(model)
        model.device = torch.device('cpu')

    for batch_i, (K1_map, label, otf3d, y) in enumerate(dataloader):
        K1_map = K1_map.to(torch.float32).to(device=model.device)
        otf3d = otf3d.to(torch.complex64).to(device=model.device)
        label = label.to(torch.float32).to(device=model.device)
        x, stage_symlosses = model(K1_map,otf3d)
        break
