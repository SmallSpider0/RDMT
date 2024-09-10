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
from data.data_transfor_mul import labelPropagation
from data.tsne_plt import tsne_plt,tsne_plt_v2,tsne_plt_v3,tsne_plt_keep
from model import MLP_Net

class coteaching(object):

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
       
       
        ##########################
        start_time = timeit.default_timer()
      
        no_select_data = train_data
        no_select_targets = noisy_targets
        no_select_clean_targets = clean_targets
        no_select_train_dataset = coteaching_Train_Dataset(no_select_data,no_select_targets,no_select_clean_targets)
        no_select_train_loader  = torch.utils.data.DataLoader(
                no_select_train_dataset, batch_size=self.args['batch_size'], shuffle=True, pin_memory=True, drop_last=False)
        noise_or_not = np.array([False] * len(no_select_data))
        noise_or_not[no_select_targets==no_select_clean_targets] = True
        model_data,model_preds,model_true = self.coteaching(no_select_train_loader, noise_or_not)
        
        
        train_data = model_data
        noisy_targets = model_preds
        clean_targets = model_true
        update_train_dataset = update_Train_Dataset(train_data, noisy_targets, clean_targets)
        update_trainloader =  torch.utils.data.DataLoader(update_train_dataset, batch_size=self.args['batch_size'], shuffle=True, pin_memory=True, drop_last=False)
        
        return train_data,noisy_targets,clean_targets,update_train_dataset,update_trainloader

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
            # evaluate models
            # test_acc1, test_acc2=self.evaluate(test_loader, mlp_1, mlp_2)
            # save results
            mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
            mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
            if epoch%5==0:
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

        # _, pred = output.topk(maxk, 1, True, True)
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