import os
from my_tool import *
import data as DT
import torch
import torch.backends.cudnn as cudnn
from model import MLP_Net
import torch.nn as nn
from zero.zero import zero
from data.train_dataset import update_Train_Dataset
from reweight.dataset import ImbalancedDatasetSampler
from zero.utils import AverageMeter
from train_model import MLP_train
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入')
    parser.add_argument('--random', type=int, default='44', help='随机数')
    parser.add_argument('--noise_rate', type=float, default='0.4', help='噪声率')
    parser.add_argument('--imb_ratio', type=float, default='20', help='不平衡率')
    parser.add_argument('--train_anom_size', type=int, default='10000', help='样本大小')
    parser.add_argument('--imb_method', type=str, default='NO', help='不平衡方式')
    parser.add_argument('--threshold', type=float, default='0.05', help='划分有无标签比例')
    parser.add_argument('--majority_label_ratio_thres', type=float, default='0.85', help='多数派标签比例阈值')
    args = parser.parse_args()
    random_seed = args.random   
    if random_seed is not None:
        set_random_seed(random_seed)

    param = {
        'random_seed': random_seed,
        'device': torch.device('cpu'),
        # 'data' :'DoH',
        # 'root' : 'd:/DoHBrw-2020',
        # 'dataset':['normalized_DoHBrw'],
        'data' :'CIC17',
        'root' : 'd:/CIC17_18',
        'dataset':['CIC_17_day1',
                    'CIC_17_day2',
                    'CIC_17_day3',
                    'CIC_17_day4',
                    'CIC_17_day5'],
        
        'noise_type' : 'sym', #对称噪声symmetric 非对称噪声asymmetric
        'imb_type' : 'ratio',
        'num_class' : 2,
        'imb_ratio' : args.imb_ratio, #数据不平衡比率, 
        'noise_rate' :args.noise_rate, #噪声率,
        'train_norm_size' : args.train_anom_size, #训练样本数量 
        'batch_size' : 512, 
        'use_scl':False,
        'threshold' : 0.35, 
        'imb_method' : args.imb_method, #除了加权还有 'CB''WS''NO'
        'epoch' : 100,
        'finally_epoch' : 100,
        'dist_alignment' : False,
        'ratio' : args.threshold, # 划分有无标签比例
        'warmup':10, #预热轮次区分有无标签
        'dataset_origin' : 'DOH',
        'use_true_distribution' : False,    # 是否使用真实分布
        'clean_method' : 'confidence', 
        'majority_label_ratio_thres' : args.majority_label_ratio_thres, # 多数派标签比例阈值
        'input_dim' :0,
        'num_iter_per_epoch':400,
        'forget_rate' : 0.1, # 遗忘率
        'num_gradual' : 5, #控制遗忘
        'exponent' : 0.5, #控制遗忘
    }

    gan_param = {
        'debug_feature_log_freq': 100, #生成器生成样本的频率
        'batch_size': 128,
        'epoch': 100,
        }
    
    modle_param = {
        'optimizer' : 'adam',
        'lr' : 0.001,
        'weight_decay' : 0.0001,
        'nesterov' : True,
        'momentum' : 0.9,
    }

    if torch.cuda.is_available():
            param['device'] = torch.device('cuda')
            cudnn.deterministic = True
            cudnn.benchmark = True
    else:
        param['device'] = torch.device('cpu')
        # args.gpu_index = -1

    #数据处理
    dataset_train, dataset_test, train_data, noisy_targets, clean_labels = DT.get_data.getdataset(param['root'], param['dataset'], param['train_norm_size'], 
                                                                                                param['noise_type'], param['noise_rate'], param['imb_type'], 
                                                                                               param['imb_ratio'],param['data'])
    train_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size=param['batch_size'], shuffle=True, pin_memory=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
                dataset_test, batch_size=param['batch_size'], shuffle=False, pin_memory=True, drop_last=False)
    print('训练样本size：{}，   测试样本size：{}'.format(len(train_data), len(dataset_test.data)))
    #####################################################################################
    input_dim = len(train_data[0])
    param['input_dim'] = input_dim
    model = MLP_Net(input_dim, [100, 100,32, param['num_class']], batch_norm=nn.BatchNorm1d, use_scl=param['use_scl'])
    if modle_param['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), modle_param['lr'], weight_decay=modle_param['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), modle_param['lr'], momentum=modle_param['momentum'],
                                    weight_decay=modle_param['weight_decay'], nesterov=modle_param['nesterov'])
    model_2 = model
    milestones = [10, 60, 90]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                milestones=milestones, gamma=0.3, last_epoch=-1) 
    # useless
    dist = [0.14, 0.15, 0.15, 0.12, 0.15, 0.09, 0.01, 0.12, 0.03, 0.02, 0.01, 0.01]

    zero = zero(model_1=model,model_2 = model_2, optimizer=optimizer, scheduler=lr_scheduler, \
                                dist=dist, param=param, modle_param=modle_param)

    train_data,de_noisy_targets,clean_labels,update_train_dataset,update_trainloader = zero.run(train_data, clean_labels,noisy_targets,train_loader,test_loader)
    #################################################################################
    imb_labeled_sampler =  ImbalancedDatasetSampler(update_train_dataset, num_class=param['num_class'])
    imb_labeled_loader = torch.utils.data.DataLoader(dataset=update_train_dataset, batch_size=128, shuffle=False,
                        pin_memory=True, sampler=imb_labeled_sampler,
                            drop_last=True)

    # update criterion
    if param['imb_method'] == 'CB':
        cls_num_list = imb_labeled_sampler.label_to_count       
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(param['device'])
    elif param['imb_method'] == 'WS':
        cls_num_list = imb_labeled_sampler.label_to_count   
        total_samples = 0
        for i in cls_num_list:
            total_samples += i  
        per_cls_weights = [total_samples / cls_num for cls_num in cls_num_list]
        # 将权重标准化，使它们的和为类别数量
        per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float32)
        per_cls_weights = per_cls_weights / per_cls_weights.sum() * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(param['device'])
    elif param['imb_method'] == 'NO':
        per_cls_weights = torch.FloatTensor([1,1]).to(param['device'])
    elif param['imb_method'] == 'SD': 
        per_cls_weights = torch.FloatTensor([0.5,3]).to(param['device'])
    mlp_net = MLP_train(model=model, optimizer=optimizer, scheduler=lr_scheduler, \
                             param=param, modle_param=modle_param)
    mlp_net.train(update_trainloader,test_loader,per_cls_weights)

    