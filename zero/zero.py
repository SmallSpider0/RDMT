import numpy as np
import timeit
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from zero.utils import AverageMeter, predict_dataset_softmax, get_labeled_dist
from zero.utils import debug_label_info, debug_unlabel_info, debug_real_label_info, debug_real_unlabel_info, debug_threshold
from zero.utils import refine_pesudo_label, update_proto, init_prototype, dynamic_threshold
from data.train_dataset import Train_Dataset,Labeled_Dataset,Unlabeled_Dataset,update_Train_Dataset,coteaching_Train_Dataset
from torch.utils.data import DataLoader
from zero.losses import LDAMLoss, SupConLoss, ce_loss ,DebiasedSupConLoss,loss_coteaching
from zero.ensemble_cluster import *
from my_tool import *
from model import MLP_Net

class zero(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['param']
        self.model_args = kwargs['modle_param']
        # load model, ema_model, optimizer, scheduler
        self.model_1 = kwargs['model_1'].to(self.args['device'])
        self.model_2 = kwargs['model_2'].to(self.args['device'])
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        
        self.train_model_tag = 1

        self.init_threshold = self.args['threshold']
        # self.log_dir = kwargs['logdir']
        # # tensorboard writer
        # self.writer = SummaryWriter(log_dir=self.log_dir)
        self.update_cnt = 0
        # Distribution of noisy data
        self.dist = kwargs['dist']
        self.cls_threshold = np.array(self.dist) * self.args['threshold'] / max(self.dist)
        self.cls_threshold = torch.Tensor(self.cls_threshold).to(self.args['device']) 

        if self.args['imb_method'] == 'resample' or self.args['imb_method'] == 'mixup':
           self.criterion = nn.CrossEntropyLoss().to(self.args['device'])

        self.per_cls_weights = None
        self.unlabel_per_cls_weights = None
        # SupConLoss
        self.criterion_con = DebiasedSupConLoss(temperature=0.07)
        self.threshold = None

    #主要运行函数
    def run(self, train_data, clean_targets, noisy_targets, trainloader, testloader):
        self.train_num = clean_targets.shape[0]
        best_acc = 0.0
        # dist_alignment or not
        if self.args['dist_alignment']:
            self.prev_labels = torch.full(
                [self.args.dist_alignment_batches, self.args['num_class']], 1 / self.args['num_class'], device=self.args['device'])
            self.prev_labels_idx = 0
       
        #聚类
        Clu_X, Clu_label,sil_set, my_Clusters = Ensemble_Cluster(X=train_data, y=clean_targets,score=None, clf_max_cluster_num=70)
        tsne_2_flag = True
        
        M = construct_matrix(sil_set=sil_set, Clu_X=Clu_X, Clu_label=Clu_label)
        ######## 预热训练部分 ########
        for i in range(self.args['warmup']):
            self.warmup(i, trainloader)
        ##########################
        start_time = timeit.default_timer()
        labeled_dataset,unlabeled_dataset,labeled_indexs, unlabeled_indexs = self.update_dataset(trainloader, train_data, clean_targets, noisy_targets,my_Clusters)
        self.train_model_tag = 2

        total_cluster_index = []
        for clustered_sample_idxs in my_Clusters:
            total_cluster_index.extend(clustered_sample_idxs)
        y_in_clustered_order = noisy_targets[np.array(total_cluster_index)]
        # 算法计算传播
        s_for_label = np.zeros(len(train_data))
        s_for_label[labeled_indexs] = 1
        s_for_label = s_for_label[np.array(total_cluster_index)]
        labeled_X, labeled_y, keep_mask,keep_mask_co = label_propagation(self.args,train_data, s_for_label, my_Clusters, Clu_label, M,
                                                            y_in_clustered_order=y_in_clustered_order)
        
        ###################################################################
        no_select_data = train_data[keep_mask_co==False]
        no_select_targets = noisy_targets[keep_mask_co==False]
        no_select_clean_targets = clean_targets[keep_mask_co==False]
        no_select_train_dataset = coteaching_Train_Dataset(no_select_data,no_select_targets,no_select_clean_targets)
        no_select_train_loader  = torch.utils.data.DataLoader(
                no_select_train_dataset, batch_size=self.args['batch_size'], shuffle=True, pin_memory=True, drop_last=False)
        noise_or_not = np.array([False] * len(no_select_data))
        noise_or_not[no_select_targets==no_select_clean_targets] = True
        model_data,model_preds,model_true = self.coteaching(no_select_train_loader, noise_or_not)
        no_select_data_label_anomidx = np.where(model_preds==1)
        select_anom_data = model_data[no_select_data_label_anomidx]
        select_anom_label = model_preds[no_select_data_label_anomidx]
        select_anom_clean_label = model_true[no_select_data_label_anomidx]
       
        norm_idxs = clean_targets[keep_mask_co] == 0
        same_idxs = labeled_y == clean_targets[keep_mask_co]
        print('[标签传播]  正确比例：0：{:5d}/{:5d} ={:.2f}  \t1：{:5d}/{:5d} ={:.2f}'.format(
            np.sum(norm_idxs & same_idxs), np.sum(norm_idxs),
            np.sum(norm_idxs & same_idxs) / np.sum(norm_idxs) * 100,
            np.sum(~norm_idxs & same_idxs), np.sum(~norm_idxs),
            np.sum(~norm_idxs & same_idxs) / np.sum(~norm_idxs) * 100
        ))
        print('\t\t正确比例：0：{:5d}/{:5d} ={:.2f}'.format(
            np.sum(same_idxs), len(same_idxs),
            np.sum(same_idxs) / len(same_idxs) * 100
        ))
        ######################################################
        train_data = labeled_X
        noisy_targets = labeled_y
        clean_targets = clean_targets[keep_mask_co==True]
        train_data = np.concatenate((train_data,select_anom_data),axis=0)
        noisy_targets = np.concatenate((noisy_targets,select_anom_label),axis=0)
        clean_targets = np.concatenate((clean_targets,select_anom_clean_label),axis=0)
        update_train_dataset = update_Train_Dataset(train_data, noisy_targets, clean_targets)
        update_trainloader =  torch.utils.data.DataLoader(update_train_dataset, batch_size=self.args['batch_size'], shuffle=True, pin_memory=True, drop_last=False)
        
        return train_data,noisy_targets,clean_targets,update_train_dataset,update_trainloader

    #提前预热训练划分 有标记和无标记数据集
    def warmup(self, epoch, trainloader):
        self.model_1.train()
        batch_idx = 0
        losses = AverageMeter()
        for i, (x, y, _) in  enumerate(trainloader):
            x = x.to(self.args['device'])
            y = y.to(self.args['device'])
            logits = self.model_1(x)
            
            self.optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), len(logits))

            batch_idx += 1

        print('Epoch [%3d/%3d] Loss: %.2f' % (epoch, self.args['epoch'], losses.avg))
    
    def train_model(self, epoch, trainloader):
        self.model_2.train()
        batch_idx = 0
        losses = AverageMeter()
        for i, (x, y, _) in  enumerate(trainloader):
            x = x.to(self.args['device'])
            y = y.to(self.args['device'])
            logits = self.model_2(x)
            
            self.optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), len(logits))

            batch_idx += 1

        print('Epoch [%3d/%3d] Loss: %.2f' % (epoch, self.args['epoch'], losses.avg))

    #训练基础模型
    def eval(self, testloader, eval_model, epoch):
        eval_model.eval()  # Change model to 'eval' mode.
        correct = 0
        total = 0

        # return the class-level accuracy
        model_preds = []
        model_true = []

        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                x = x.to(self.args['device'])
                y = y.to(self.args['device'])
                logits = eval_model(x)

                outputs = F.softmax(logits, dim=1)
                _, pred = torch.max(outputs.data, 1)

                total += y.size(0)
                correct += (pred.cpu() == y.cpu().long()).sum()

                # add pred1 | labels
                model_preds.append(pred.cpu())
                model_true.append(y.cpu().long())

        model_preds = np.concatenate(model_preds, axis=0)
        model_true = np.concatenate(model_true, axis=0)

        evaluate_true_pred_label(model_true, model_preds)
        cm = confusion_matrix(model_true, model_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()

        acc = 100 * float(correct) / float(total)
        # print(class_acc)
        print('Epoch [%3d/%3d] Test Acc: %.2f%%' %(epoch, self.args['epoch'], acc))
        gap_cls = int(math.ceil(self.args['num_class'] / 2))

        return acc, class_acc
    
    def update_dataset(self, train_loader, train_data, clean_targets, noisy_targets,my_Clusters=None):
        if self.train_model_tag == 1:
            soft_outs, preds = predict_dataset_softmax(train_loader, self.model_1, self.args['device'], self.train_num)
        else:
            soft_outs, preds = predict_dataset_softmax(train_loader, self.model_2, self.args['device'], self.train_num)
        labeled_indexs, unlabeled_indexs = self.splite_confident(soft_outs, clean_targets, noisy_targets,self.args['ratio'],my_Clusters)
        labeled_dataset = Labeled_Dataset(train_data[labeled_indexs], noisy_targets[labeled_indexs],\
                                          clean_targets[labeled_indexs])
        unlabeled_dataset = Unlabeled_Dataset(train_data[unlabeled_indexs], noisy_targets[unlabeled_indexs], \
                                                clean_targets[unlabeled_indexs])
        labeled_num, unlabeled_num = len(labeled_indexs), len(unlabeled_indexs)

        # print confident set clean-ratio
        noisy_norm_idx = np.where(clean_targets[labeled_indexs]==0)
        noisy_anom_idx = np.where(clean_targets[labeled_indexs]==1)
        clean_norm_num = np.sum(noisy_targets[noisy_norm_idx]==clean_targets[noisy_anom_idx])
        clean_anom_num = np.sum(noisy_targets[noisy_norm_idx]==clean_targets[noisy_anom_idx])

        clean_num = np.sum(noisy_targets[labeled_indexs]==clean_targets[labeled_indexs])
        clean_ratio = clean_num * 1.0 / labeled_num

        noise_label = noisy_targets[labeled_indexs]
        clean_label = clean_targets[labeled_indexs]
        print(f' clean_norm_num is {clean_norm_num}')
        print(f' clean_anom_num is {clean_anom_num}')
        print('Labeled data clean ratio is %.4f' %clean_ratio)
        cls_precision = debug_label_info(noise_label, clean_label, self.args['num_class'])
        print(f'clean_anom_ratio is {cls_precision[0]}, clean_anom_num is {cls_precision[1]}')
        return labeled_dataset,unlabeled_dataset,labeled_indexs, unlabeled_indexs

    #划分有标记和无标记
    def splite_confident(self, outs, clean_targets, noisy_targets,ratios,my_Clusters=None):
        # 存储已标记的索引
        labeled_indexs = []
        # 存储未标记的索引
        unlabeled_indexs = []

        if not self.args['use_true_distribution']:
            # 如果使用的是置信度方法
            if self.args['clean_method'] == 'confidence':
                # 遍历所有类别
                for cls in range(self.args['num_class']):
                # 获取当前类别的索引
                    idx = np.where(noisy_targets==cls)[0]
                    # 获取当前类别的损失值
                    loss_cls = outs[idx]
                    # 对损失值进行排序
                    sorted, indices = torch.sort(loss_cls, descending=False)
                    # 选择前5%的索引作为已标记的索引
                    select_num = int(len(indices) * ratios)
                    # 遍历排序后的索引
                    for i in range(len(indices)):
                    # 如果索引小于选择数量，则添加到已标记的索引列表中
                        if i < select_num:
                            labeled_indexs.append(idx[indices[i].item()])
                    # 否则，添加到未标记的索引列表中
                        else:
                            unlabeled_indexs.append(idx[indices[i].item()])
            elif self.args['clean_method'] == 'Clusters':
                # 遍历所有类别
                for i_cluster, clustered_sample_idxs in enumerate(my_Clusters):
                    # 获取当前类别的损失值
                    loss_cls = outs[clustered_sample_idxs]
                     # 对损失值进行排序
                    sorted, indices = torch.sort(loss_cls, descending=False)
                    # 选择前ratios的索引作为已标记的索引
                    select_num = int(len(indices) * ratios)
                    if select_num == 0:
                        select_num = 1
                    # 遍历排序后的索引
                    for i in range(len(indices)):
                        # 如果索引小于选择数量，则添加到已标记的索引列表中
                        if i < select_num:
                            labeled_indexs.append(clustered_sample_idxs[indices[i].item()])
                        # 否则，添加到未标记的索引列表中
                        else:
                            unlabeled_indexs.append(clustered_sample_idxs[indices[i].item()])
            elif self.args['clean_method'] == 'All':
                # 遍历所有类别
                loss_cls = outs
                # 对损失值进行排序
                sorted, indices = torch.sort(loss_cls, descending=False)
                # 选择前15%的索引作为已标记的索引
                select_num = int(len(indices) * ratios)
                # 遍历排序后的索引
                for i in range(len(indices)):
                # 如果索引小于选择数量，则添加到已标记的索引列表中
                    if i < select_num:
                        labeled_indexs.append(indices[i].item())
                # 否则，添加到未标记的索引列表中
                    else:
                        unlabeled_indexs.append(indices[i].item())
        else:
            sz = noisy_targets.shape[0]
            for cls in range(self.args['num_class']):
                idx = np.where(noisy_targets == cls)[0]
                select_num = int(sz * ratios * self.dist[cls])
                cnt = 0
                for i in range(len(idx)):
                    if noisy_targets[idx[i]] == clean_targets[idx[i]]:
                       cnt += 1
                       if cnt <= select_num:
                          labeled_indexs.append(idx[i])
                       else:
                          unlabeled_indexs.append(idx[i])
                    else:
                         unlabeled_indexs.append(idx[i])

        return labeled_indexs, unlabeled_indexs
    ##############################找异常点部分#################################################################
    def adjust_learning_rate(self,optimizer, epoch):
            mom1 = 0.9
            mom2 = 0.1
            learning_rate = self.model_args['lr'] 
            alpha_plan = [learning_rate] * self.args['epoch']
            beta1_plan = [mom1] * self.args['epoch']
            for param_group in optimizer.param_groups:
                param_group['lr']=alpha_plan[epoch]
                param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
    def coteaching(self, train_loader, noise_or_not):
        input_dim = self.args['input_dim']
        mlp_1 = MLP_Net(input_dim, [100, 100,32, self.args['num_class']], batch_norm=nn.BatchNorm1d, use_scl=self.args['use_scl'])
        mlp_2 = MLP_Net(input_dim, [100, 100,32, self.args['num_class']], batch_norm=nn.BatchNorm1d, use_scl=self.args['use_scl'])
        optimizer1 = torch.optim.Adam(mlp_1.parameters(), self.model_args['lr'], weight_decay=self.model_args['weight_decay'])
        optimizer2 = torch.optim.Adam(mlp_2.parameters(), self.model_args['lr'], weight_decay=self.model_args['weight_decay'])
        mean_pure_ratio1=0
        mean_pure_ratio2=0
        epoch=0
        train_acc1=0
        train_acc2=0
        mlp_1.to(self.args['device'])
        mlp_2.to(self.args['device'])
        for epoch in range(1, self.args['epoch']):
            # train models
            mlp_1.train()
            self.adjust_learning_rate(optimizer1, epoch)
            mlp_2.train()
            self.adjust_learning_rate(optimizer2, epoch)
            train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=self.train(train_loader, epoch, mlp_1, optimizer1, mlp_2, optimizer2,noise_or_not)

            mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
            mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
            if epoch%5==1:
                print('Epoch [%d/%d]  Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1,self.args['epoch'], mean_pure_ratio1, mean_pure_ratio2))
        model_data,model_preds,model_true = self.co_eval(train_loader,mlp_1,epoch)
        self.co_eval(train_loader,mlp_2,epoch)
        return model_data,model_preds,model_true
        ###预测阶段
        
    
    def accuracy(logit, target, topk=1):
        """Computes the precision@k for the specified values of k"""
        
        # output = FF.softmax(logit, dim=1)
        prob, pred = torch.max(F.softmax(logit, dim=1), dim=1)
        maxk = max(topk)
        batch_size = target.size(0)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    def evaluate(test_loader, model1, model2):
        model1.eval()    # Change model to 'eval' mode.
        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            logits1 = model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        model2.eval()    # Change model to 'eval' mode 
        correct2 = 0
        total2 = 0
        for images, labels, _ in test_loader:
            logits2 = model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()
    
        acc1 = 100*float(correct1)/float(total1)
        acc2 = 100*float(correct2)/float(total2)
        return acc1, acc2


    def train(self,train_loader,epoch, model1, optimizer1, model2, optimizer2,noise_or_not):
        
        pure_ratio_list=[]
        pure_ratio_1_list=[]
        pure_ratio_2_list=[]
        # define drop rate schedule
        rate_schedule = np.ones(self.args['epoch'])*self.args['forget_rate']
        rate_schedule[:self.args['num_gradual']] = np.linspace(0, self.args['forget_rate']**self.args['exponent'], self.args['num_gradual'])
   
        train_total=0
        train_correct=0 
        train_total2=0
        train_correct2=0 

        for i, (feature, labels, indexes,_) in enumerate(train_loader):
            ind=indexes.cpu().numpy().transpose()
            feature = feature.to(self.args['device'])
            labels = labels.to(self.args['device'])
            
            if i>self.args['num_iter_per_epoch']:
                break
    
            # Forward + Backward + Optimize
            logits1=model1(feature)
            target = labels
            ####### accuracy ##########
            output = F.softmax(logits1, dim=1)
            topk = (1)
            # maxk = max(topk)
            maxk = topk
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            # for k in topk:
            # correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            ###############################
            prec1 = res[0]
            train_total+=1
            train_correct+=prec1

            logits2 = model2(feature)
            ####### accuracy ##########
            output = F.softmax(logits1, dim=1)
            topk = (1)
            # maxk = max(topk)
            maxk = topk
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            # for k in topk:
            # correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            ###############################
            prec2= res[0]
            train_total2+=1
            train_correct2+=prec2
            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
            pure_ratio_1_list.append(100*pure_ratio_1)
            pure_ratio_2_list.append(100*pure_ratio_2)

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            # if i % 5 == 0:
            #     print ('Epoch [%d/%d], Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
            #         %(epoch+1, self.args['epoch'], prec1, prec2, loss_1, loss_2, np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

        train_acc1=float(train_correct)/float(train_total)
        train_acc2=float(train_correct2)/float(train_total2)
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list
        ###################################################################################


    def co_eval(self, testloader, eval_model, epoch):
        eval_model.eval()  # Change model to 'eval' mode.
        correct = 0
        total = 0

        # return the class-level accuracy
        model_data = []
        model_preds = []
        model_true = []

        with torch.no_grad():
            for i, (x,_,_,y) in enumerate(testloader):
                x = x.to(self.args['device'])
                y = y.to(self.args['device'])
                logits = eval_model(x)

                outputs = F.softmax(logits, dim=1)
                _, pred = torch.max(outputs.data, 1)

                total += y.size(0)
                correct += (pred.cpu() == y.cpu().long()).sum()

                # add pred1 | labels
                model_data.append(x.cpu())
                model_preds.append(pred.cpu())
                model_true.append(y.cpu().long())

        model_data = np.concatenate(model_data, axis=0)
        model_preds = np.concatenate(model_preds, axis=0)
        model_true = np.concatenate(model_true, axis=0)

        TN, FP, FN, TP = evaluate_true_pred_label(model_true, model_preds)
        cm = confusion_matrix(model_true, model_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()

        acc = 100 * float(correct) / float(total)
        # print(class_acc)
        print('Epoch [%3d/%3d] Test Acc: %.2f%%' %(epoch, self.args['epoch'], acc))
        gap_cls = int(math.ceil(self.args['num_class'] / 2))

        return model_data,model_preds,model_true