import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader

class Test_Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = np.array(data,dtype=np.float32)
        self.targets = np.array(labels,dtype= int)
        self.length = len(self.targets)

        anom_idx = np.where(np.array(self.targets)!=0)
        self.targets[anom_idx] = 1

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        feat, target = self.data[index], self.targets[index]

        if self.transform is not None:
            feat = self.transform(feat)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feat, target

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets