import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.noise_data import noisify
import torch

class Train_Dataset(Dataset):

    def __init__(self, data, label, transform = None, num_classes = 10,
                 noise_type='symmetric', noise_rate=0.5, select_class=-1):

        self.num_classes = num_classes

        self.train_data = data
        self.train_labels = label

        self.gt = self.train_labels.copy()
        self.gt = np.array(self.gt)
        anom_idx = np.where(self.gt!=0)
        self.gt[anom_idx] = 1

        self.transform = transform
        self.train_noisy_labels = self.train_labels.copy()
        self.noise_or_not = np.array([True for _ in range(len(self.train_labels))])
        self.P = np.zeros((num_classes, num_classes))

        if noise_type !='none':
            # noisify train data
            self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
            
            self.train_noisy_labels, self.actual_noise_rate, self.P = noisify(train_labels=self.train_labels,
                            noise_type=noise_type, noise_rate=noise_rate, random_state=0, nb_classes=self.num_classes,
                            select_class=select_class)
            
            self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
            self.train_noisy_labels = np.array(self.train_noisy_labels)
            anom_idx = np.where(np.array(self.train_noisy_labels)!=0)
            self.train_noisy_labels[anom_idx] = 1

            _train_labels=[i[0] for i in self.train_labels]
            _train_labels = np.array(_train_labels)
            anom_idx = np.where(np.array(_train_labels)!=0)
            _train_labels[anom_idx] = 1
            self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
            

    def __getitem__(self, index):

        feat, gt = self.train_data[index], int(self.train_noisy_labels[index])
        if self.transform is not None:
            feat = self.transform(feat)
        return feat, gt,index

    def __len__(self):
        return len(self.train_data)
    

class Labeled_Dataset(Dataset):
    def __init__(self, data, noise_labels,gts):
        self.data = np.array(data,dtype=np.float32)
        self.targets = np.array(noise_labels, dtype = int)
        self.length = len(self.targets)
        self.dim = self.data.shape[1]
        self.gts = np.array(gts,dtype=np.float32)

        # self.weak_ratio = 0.1

    def __getitem__(self, index):
        # 获取数据和噪声目标
        img, target,gt = self.data[index], self.targets[index], self.gts[index]
       
        return img, target, 

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets


class Unlabeled_Dataset(Dataset):
    def __init__(self, data, labels, gts):
        self.data = np.array(data)
        self.targets = np.array(labels)
        self.gt = np.array(gts)
        self.length = self.data.shape[0]
        self.dim = self.data.shape[1]

        self.aug_ratio = [0.1, 0.2]

    def __getitem__(self, index):
        data, target, gt = self.data[index], self.targets[index], self.gt[index]
        

        return data, target, gt

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets
    
class update_Train_Dataset(Dataset):
    def __init__(self, data, labels, gts):
        self.data = torch.tensor(np.array(data))
        self.targets = torch.tensor(np.array(labels))
        self.gt = torch.tensor(np.array(gts))
        self.length = len(self.targets)
    def __getitem__(self, index):
        data, target, gt = self.data[index], self.targets[index], self.gt[index]
        
        return data, target, gt

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets
    
class coteaching_Train_Dataset(Dataset):
    def __init__(self, data, labels, gts):
        self.data = np.array(data)
        self.targets = np.array(labels)
        self.gt = np.array(gts)
        self.length = len(self.targets)
    def __getitem__(self, index):
        data, target, gt = self.data[index], self.targets[index], self.gt[index]
        
        return data, target, index,gt

    def __len__(self):
        return self.length

    def getData(self):
        return self.data, self.targets