import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ce_loss(logits, targets, use_hard_labels=True, weight=None, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, weight=weight, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        if weight == None:
           nll_loss = torch.sum(-targets * log_pred, dim=1)
        else:
           nll_loss = torch.sum(-targets * log_pred * weight, dim=1)
        return nll_loss

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.device = device

    def forward(self, x, target, reduction):
        index = torch.zeros_like(x, dtype=torch.uint8).to(self.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        if reduction != None:
           return F.cross_entropy(self.s*output, target, weight=self.weight)
        else:
           return F.cross_entropy(self.s*output, target, weight=self.weight, reduction=reduction)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)    ## make adjacency matrix with diagnoals = 1

        contrast_count = features.shape[1]  ## 1 ??
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  ## removed the unused dimension

        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        # compute correlations
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask   ## final adjacency matrix

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos_numer = (mask * log_prob).sum(1)
        mean_log_prob_pos_denom = mask.sum(1)
        temp = mask.sum(1)
        nonzero_num = temp.shape[0] - torch.sum(temp == 0)
        mean_log_prob_pos_numer = mean_log_prob_pos_numer[temp != 0]
        mean_log_prob_pos_denom = mean_log_prob_pos_denom[temp != 0]
        mean_log_prob_pos = mean_log_prob_pos_numer / mean_log_prob_pos_denom

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.view(anchor_count, nonzero_num).mean()
        #loss = torch.mean(loss)
        return loss



class DebiasedSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(DebiasedSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, biased_label=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None and biased_label is None:
            label_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            label_mask = torch.eq(labels, labels.T)
            if biased_label is not None:
                bias_mask = torch.logical_or(biased_label, biased_label.T)
                #bias_mask = torch.ne(biased_label, biased_label.T)
                mask = label_mask & bias_mask
            else:
                mask = label_mask
            mask = mask.float().to(device)
        else:
            raise ValueError

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        valid_idx = mask.sum(1) != 0
        mask = mask[valid_idx]
        logits_mask = logits_mask[valid_idx]

        # compute log_prob
        logits = logits[valid_idx]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')

        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1).long().squeeze())
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        targets = torch.argmax(targets, dim=1)
        loss_weight = torch.argmax(loss_weight, dim=1)

        # print(logits.shape,targets.shape)
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss
    
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    t = t.long()
    t_one_hot = F.one_hot(t, num_classes=2)
    t_one_hot = t_one_hot.float()
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_1_numpy = loss_1.cpu().detach().numpy()
    ind_1_sorted = np.argsort(loss_1_numpy)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    loss_2_numpy = loss_2.cpu().detach().numpy()
    ind_2_sorted = np.argsort(loss_2_numpy)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    # loss_1_update = F.mse_loss(y_1[ind_2_update], t[ind_2_update])
    # loss_2_update = F.mse_loss(y_2[ind_1_update], t[ind_1_update])
    loss_1_update = F.l1_loss(y_1[ind_2_update], t_one_hot[ind_2_update],reduction='mean')
    loss_2_update = F.l1_loss(y_2[ind_1_update], t_one_hot[ind_1_update],reduction='mean')

    # return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
    return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2     


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss
