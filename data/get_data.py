import numpy as np
import pandas as pd
import math
from data.train_dataset import Train_Dataset
from data.test_dataset import Test_Dataset

def getdataset(root, dataset, trainsize, noise_type, noise_rate, imb_type,imb_ratio,data):
    # noise-data
    ############## DOH划分数据集 ##############
    if data == 'DoH':
        label_uniq_list = ['Benign']
        all_data = np.empty((0,52))
        for i,dataname in enumerate(dataset):
            add_data = np.array(pd.read_csv(root+ '/' + dataname + '.csv'))
            all_data = np.concatenate([all_data,add_data],axis=0)
        all_data_labels = all_data[:,-1]
        all_data = all_data[:,:-1]
        my_dict = {'Benign': 0,'NonDoH':0 ,'Malicious': 1}
        for i in range(len(all_data_labels)):
            all_data_labels[i] = my_dict[all_data_labels[i]]
        no_infi_idx = np.where(all_data_labels!=6)[0]
        all_data = all_data[no_infi_idx]
        all_data_labels = all_data_labels[no_infi_idx]
        num_classes = len(np.unique(all_data_labels))
    ############################################
    ##############  CIC17划分数据集 ###############
    if data == "CIC17":
        label_uniq_list = ['Benign']
        all_data = np.empty((0,78))
        for i,dataname in enumerate(dataset):
            add_data = np.array(pd.read_csv(root + '/' + dataname + '.csv'))
            all_data = np.concatenate([all_data,add_data],axis=0)
        all_data_labels = all_data[:,-1]
        all_data = all_data[:,:-1]
        
        my_dict = {'Benign': 0, 'FTP-BruteForce': 1, 'SSH-BruteForce': 1, 'DoS Slowloris': 2, 'DoS SlowHTTPTest': 2, 'DoS Hulk': 2, 
            'DoS GoldenEye': 2, 'Heartbleed': 2, 'Web-BruteForce': 3, 'Web Attack - XSS': 3, 'Sql Injection': 3, 
                'Bot': 4, 'PortScan': 5, 'DDoS': 6,'Infiltration': 7 }
        for i in range(len(all_data_labels)):
            all_data_labels[i] = my_dict[all_data_labels[i]]
        no_infi_idx = np.where(all_data_labels!=7)[0]
        all_data = all_data[no_infi_idx]
        all_data_labels = all_data_labels[no_infi_idx]
        num_classes = len(np.unique(all_data_labels))
    #---------------------------------------------------------
    if data == "CIC18":
        label_uniq_list = ['Benign']
        all_data = np.empty((0,78))
        for i,dataname in enumerate(dataset):
            add_data = np.array(pd.read_csv(root + '/' + dataname + '.csv'))
            all_data = np.concatenate([all_data,add_data],axis=0)
        all_data_labels = all_data[:,-1]
        all_data = all_data[:,:-1]
        
        my_dict = {'Benign': 0, 'FTP-BruteForce': 1, 'SSH-BruteForce': 1, 'DoS Slowloris': 2, 'Brute Force -XSS': 1,
                    'DoS SlowHTTPTest': 2, 'DoS Hulk': 2, 'DoS GoldenEye': 2, 
                    'Web-BruteForce': 3, 'Sql Injection': 3, 
                'Bot': 4, 'DDOS attack-HOIC': 5,'DDOS attack-LOIC-UDP': 5, 'Infiltration': 6 }
        for i in range(len(all_data_labels)):
            all_data_labels[i] = my_dict[all_data_labels[i]]
        no_infi_idx = np.where(all_data_labels!=6)[0]
        all_data = all_data[no_infi_idx]
        all_data_labels = all_data_labels[no_infi_idx]
        num_classes = len(np.unique(all_data_labels))
    ##############################################
 
    print('-'*12 + f'对应标签数量：{num_classes}' + '-'*12)
    print('-'*32)
    # 判断是否需要进行imbalanced
    if imb_type != 'none':
        # 初始化类别的数量列表
        cls_num_list = []
        num_cls = []
        for i in range(num_classes):
            num_cls.append(np.sum(all_data_labels==i))
        # 如果imb_type为step，则按照step的方式进行
        if imb_type == 'step':
            # 计算每个类别的数量
            for i in range(int(math.ceil(num_classes / 2))):
                cls_num_list.append(trainsize)
            # 计算每个类别的数量
            for i in range(int(num_classes / 2)):
                cls_num_list.append(int(trainsize * imb_ratio))
        # 如果imb_type不为step，则按照ratio的方式进行
        elif imb_type == 'step_anom':
            # 计算每个类别的数量
            for i in range(int(num_classes)):
                if i==0 :
                    cls_num_list.append(int(trainsize*imb_ratio*(num_classes-1)))
                else:
                    cls_num_list.append(int(trainsize))
        elif imb_type == 'ratio':
            # 计算每个类别的数量
            for i in range(int(num_classes)):
                if i==0 :
                    cls_num_list.append(int(trainsize*imb_ratio/(imb_ratio+1)))
                else:
                    proportion = num_cls[i] / sum(num_cls[1:])
                    cls_num_list.append(int(trainsize*1/(imb_ratio+1)*proportion))
        else:
            # 计算每个类别的数量
            cls_num = num_classes
            for cls_idx in range(cls_num):
                num = trainsize * (imb_ratio**(cls_idx / (cls_num - 1.0)))
                cls_num_list.append(int(num))
        # 获取imbalanced的数据
        if data == 'CIC17':
            train_data, train_labels,test_data, test_labels = get_imbalanced_data_v2(num_classes, all_data, all_data_labels,  cls_num_list)
        elif data == 'DoH':
            train_data, train_labels,test_data, test_labels = get_imbalanced_data_v2(num_classes, all_data, all_data_labels,  cls_num_list)
        
    else:
        # clean-label 此处还需要修改
        # 加载clean-label
        train_data = train_data.astype(float)
        
    dataset_train = Train_Dataset(train_data, train_labels, num_classes=num_classes, noise_type=noise_type, noise_rate=noise_rate)
    dataset_test = Test_Dataset(test_data, test_labels)
    if noise_type != 'none':
        train_labels = np.array(dataset_train.train_noisy_labels, dtype=np.float32)
        clean_labels = dataset_train.gt
    return dataset_train, dataset_test, train_data, dataset_train.train_noisy_labels, clean_labels


def convert_label_to_nums(y_train,label_uniq_list,transfor_type='mul'):
    #### CIC转标签的方法 #####
    # 注意此处label_uniq_list是字符串列表，且是原始正常标签，而不是CIC的标签
    if transfor_type == 'mul':
        for i in range(len(y_train)):
            if y_train[i] == 0 or (type(y_train[i]) == type('') and y_train[i].lower() == label_uniq_list[0].lower()):
                y_train[i] = 0
            else:
                k = 1
                while k < len(label_uniq_list):
                    if label_uniq_list[k] == y_train[i]:
                        y_train[i] = k
                        break
                    k += 1

                if k >= len(label_uniq_list):
                    label_uniq_list.append(y_train[i])
                    y_train[i] = k
        return y_train.astype(int)
    if transfor_type == 'CIC_single':
        for i in range(len(y_train)):
            if y_train[i] == label_uniq_list[0]:
                y_train[i] = 0
            else:
                y_train[i] = 1
        return y_train.astype(int)
    if transfor_type == 'DoH':
        for i in range(len(y_train)):
            if y_train[i] != 'Malicious':
                y_train[i] = 0
            else:
                y_train[i] = 1
        return y_train.astype(int)
    

def get_imbalanced_data_v2(num_classes, train_data, train_labels, img_num_per_cls):
    new_data = []
    new_labels = []
    test_data = []
    test_labels = []
    for i in range(num_classes):
        idx = np.where(train_labels == i)[0]
        select_idx = np.random.choice(idx, size=img_num_per_cls[i], replace=False)
        new_data.append(train_data[select_idx, ...])
        new_labels.extend([i] * len(select_idx))
        #添加test部分
        test_select_idx = np.setdiff1d(idx,select_idx)
        test_data.append(train_data[test_select_idx, ...])
        test_labels.extend([i] * len(test_select_idx))

    new_data = np.vstack(new_data).astype(float)
    new_labels = np.array(new_labels)
    test_data = np.vstack(test_data).astype(float)
    test_labels = np.array(test_labels)
    return new_data, new_labels,test_data,test_labels
