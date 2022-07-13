from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import savemat, loadmat
import glob
import os
import numpy as np
from utils.utilis import generate_K1map
import torch

class load_qis_mat(Dataset):
    def __init__(self, path, Kt, Ks):
        self.Kt = Kt
        self.Ks = Ks
        self.path = path
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p/'**'/"*.*"),recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.mat_files = sorted([x.replace('/',os.sep) for x in f if x.split('.')[-1].lower()=='mat'])
            assert self.mat_files, f'{path}{p}No mat found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}:{e}\n')
    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        path = self.mat_files[idx]
        try:
            struct = loadmat(path)
            y  = struct['y'].astype(np.float32)
            label = struct['label'].astype(np.float32)
            otf3d = struct['otf3d'].astype(np.complex64)
            if y.shape[-1] == self.Kt:
                y = np.transpose(y,[2,0,1])
            elif y.shape[0] == self.Kt:
                pass
            else:
                raise Exception("The temperal sampling dose not match")
        except:
            print("Cannot load the .mat file")
            raise ValueError

        K1_map = generate_K1map(y,[self.Ks,self.Ks],norm=True) # range from 0-1
        return torch.from_numpy(K1_map).unsqueeze(0), torch.from_numpy(label),torch.from_numpy(otf3d),torch.from_numpy(y)



def create_dataloader_qis(path, batch_size, Kt, Ks):
    dataset = load_qis_mat(path,Kt,Ks)
    batch_size = min(batch_size,len(dataset))
    loader = torch.utils.data.DataLoader
    dataloader = loader(dataset,batch_size,shuffle=False)
    return dataloader,dataset



if __name__ == "__main__":
    path = "/Users/zhangyunping/PycharmProjects/PLholo/syn_data/data/Nz7_ppv1e-03~5e-03_dz1200um"
    dataloader, dataset = create_dataloader_qis(path,2,30,2)
    for batch_i, (K1_map, label, otf3d, y) in enumerate(dataloader):
        break# if __name__ == "__main__":
#     y_pred = torch.zeros([2,3,4,4])
#     a = 2*torch.ones([2,2])
#
#     y_true = torch.ones_like(y_pred)
#     pcc(y_pred,y_true)
