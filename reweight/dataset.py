import torch
import numpy as np

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_class=12, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * num_class
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label = label.astype(int)
            label_to_count[label] += 1
        # padding 1
        for cls in range(num_class):
            if label_to_count[cls] == 0:
                label_to_count[cls] = 1

        self.label_to_count = label_to_count
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        self.per_cls_weights = per_cls_weights
        
        # weight for each sample
        weights = []
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label = label.astype(int)
            weight = per_cls_weights[label]
            weights.append(weight)
        # weights = [per_cls_weights[self._get_label(dataset, idx)]
        #            for idx in self.indices]
        self.weights = torch.FloatTensor(weights)
        
    def _get_label(self, dataset, idx):
        labels = dataset.targets.detach().numpy()
        labels[idx] = labels[idx].astype(int)
        # dataset.targets[idx] = dataset.targets[idx].astype(int)
        return labels[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
    