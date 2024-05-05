import torch
import numpy as np
import os
from torch.utils.data import Dataset

class RelationalGamesDataset(Dataset):
    def __init__(self, data_path, task, split, transform=None):
        bin_path = f'{data_path}/{task}_{split}.bin'
        spec_path = f'{data_path}/{task}_{split}_spec.pt'

        self.imgs_memmap = np.memmap(bin_path, dtype=np.int32, mode='r')

        spec = torch.load(f'{data_path}/{task}_{split}_spec.pt')
        self.labels = spec['labels'].squeeze()
        self.data_shape = spec['shape']

        self.n, self.h, self.w, self.c = self.data_shape
        self.img_n_pixels = self.h * self.w * self.c

        self.transform = transform

    def get_img_at_idx(self, idx):
        return self.imgs_memmap[idx*self.img_n_pixels:(idx+1)*self.img_n_pixels].reshape(self.h, self.w, self.c)

    def __len__(self):
            return self.n

    def __getitem__(self, idx):
            x = self.get_img_at_idx(idx)
            x = x.transpose(2, 0, 1) / 255
            x = x.astype('float32')

            if self.transform:
                x = self.transform(x)
            return x, self.labels[idx]
