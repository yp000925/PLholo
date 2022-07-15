import torch
import torch.nn as nn
import numpy as np
import random
from torch.fft import fftn,ifftn,fftshift,ifftshift
import matplotlib.pyplot as plt
import math

def random_init(seed):
    random.seed(seed)
    np.random.seed(seed)

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

def tensor2value(tensor):
    return tensor.data.cpu().numpy()

def FT2d(a_tensor):
    if len(a_tensor.shape) == 4:
        return ifftshift(fftn(fftshift(a_tensor,dim =[2,3]),dim =[2,3]),dim =[2,3])
    elif len(a_tensor.shape) == 3:
        return ifftshift(fftn(fftshift(a_tensor,dim =[1,2]),dim =[1,2]),dim =[1,2])


def iFT2d(a_tensor):
    if len(a_tensor.shape) == 4:
        return ifftshift(ifftn(fftshift(a_tensor,dim =[2,3]),dim =[2,3]),dim =[2,3])
    elif len(a_tensor.shape) == 3:
        return ifftshift(ifftn(fftshift(a_tensor,dim =[1,2]),dim =[1,2]),dim =[1,2])

def generate_K1map(ob, block_shape, norm = False):
    Ks_m, Ks_n = block_shape
    Kt, ob_m, ob_n = ob.shape
    out_m, out_n = ob_m // Ks_m, ob_n // Ks_n
    out = np.zeros(shape=(out_m, out_n), dtype=np.float64)
    for i in range(out_m):
        for j in range(out_n):
            out[i][j] = np.sum(ob[:, i*Ks_m:(i+1)*Ks_m, j*Ks_n:(j+1)*Ks_n])
    if norm:
        return out/(Ks_m*Ks_n*Kt)
    return out

def blockfunc( ob, block_shape, func): # clear
    """
    Parameters:
        :ob: the observation of single photon imaging
        :block_shape: block shpae of this operation
        :func: the function that is applied to each block
    """

    # precompute some variables
    ob_m, ob_n = ob.shape

    # block shape
    b_m, b_n = block_shape

    # define the size of resulting image
    out_m, out_n = ob_m // b_m, ob_n // b_n

    # placeholder for the output
    out = np.zeros(shape=(out_m, out_n), dtype=np.float64)

    for i in range(out_m):
        for j in range(out_n):
            out[i][j] = func(ob[i*b_m:(i+1)*b_m, j*b_n:(j+1)*b_n])

    return out

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# %%-------------------------------------- Metrics ------------------------------------------

def accuracy(y_pred,y_true):
    """Computes the accuracy for multiple binary predictions"""
    pred = y_pred >= 0.5
    truth = y_true >= 0.5
    acc = pred.eq(truth).sum() / y_true.numel()
    return acc

def mse_loss(y_pred,y_true):
    return torch.nn.functional.mse_loss(y_pred,y_true)

def PSNR(y_pred, y_true):
    EPS = 1e-8
    mse = torch.mean((y_pred - y_true) ** 2,dim=[2,3])
    score  = -10*torch.log10(mse+EPS)
    score  = torch.mean(score)
    # score = - 10 * torch.log10(mse + EPS)
    return score

def PCC(y_pred,y_true, mean=True):
    [B,D,W,H]  = y_pred.shape
    y_pred = y_pred.view([B,D*W*H])
    y_true = y_true.view([B,D*W*H])
    mp = torch.mean(y_pred,dim =1,keepdim=True)
    mt = torch.mean(y_true,dim =1,keepdim=True)
    x_p,x_t = y_pred-mp,y_true-mt
    # std_p = torch.std(y_pred,dim=1)
    # std_t = torch.std(y_true,dim=1)

    num = torch.mean(torch.mul(x_p,x_t),dim=1)
    den = torch.norm(x_p,p=2,dim=1)*torch.norm(x_t,p=2,dim=1)
    # den = std_p*std_t
    if mean:
        return torch.mean(num/den)
    return num/den

# if __name__ == "__main__":
#     y_pred = torch.zeros([2,3,4,4])
#     a = 2*torch.ones([2,2])

# %%-------------------------------------- Display ------------------------------------------
# Plot a 3D matrix slice by slice
def plotcube(vol, fig_title, file_name):
    maxval = np.amax(vol)
    minval = np.amin(vol)
    vol = (vol - minval) / (maxval - minval)

    Nz, Nx, Ny = np.shape(vol)

    if Nz <= 10:
        img_col_n = Nz
    else:
        img_col_n = math.ceil(np.sqrt(Nz))

    img_row_n = math.ceil(Nz / img_col_n)
    image_height = 5
    fig = plt.figure(figsize=(img_col_n * image_height, image_height * img_row_n + 0.5))
    fig.suptitle(fig_title, y=1)
    img_n = 0
    for iz in range(Nz):
        img_n = img_n + 1
        ax = fig.add_subplot(img_row_n, img_col_n, img_n)
        ax.set_title("z " + str(img_n))
        slice = vol[iz, :, :]
        im = ax.imshow(slice, aspect='equal')

    fig.tight_layout()
    # plt.savefig(file_name)
    plt.show()

