from __future__ import print_function, division
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

import torch


class Sepsis_dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.len = 0
        for patient in glob(f"{root_dir}/*"):
            for bag in glob(f"{patient}/*"):
                self.image_paths.append(bag)
                self.len += 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs = torch.Tensor().to()
        label = self.image_paths[idx].split("_")[-2].split("\\")[1]
        if label == "Escherichia.coli":
            label = 0
        elif label == "Klebsiella.pneumoniae":
            label = 1
        else:
            label = 2
        # print(self.image_paths[idx])
        for img in glob(f"{self.image_paths[idx]}/*"):
            img = Image.open(img).convert('RGB')
            img = self.transform(img)
            img = img.to(torch.float)
            img = img.unsqueeze(0)
            imgs = torch.cat([imgs, img], axis=0)
        return imgs, label
