import numpy as np
import timeit
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from zero.utils import AverageMeter, predict_dataset_softmax
from zero.ensemble_cluster import *
from my_tool import *
from zero.losses import FocalLoss

class MLP_train(object):
    def __init__(self, *args, **kwargs):
            self.args = kwargs['param']
            self.model_args = kwargs['modle_param']
            # load model, ema_model, optimizer, scheduler
            self.model = kwargs['model'].to(self.args['device'])
            self.optimizer = kwargs['optimizer']
            self.scheduler = kwargs['scheduler']

            self.init_threshold = self.args['threshold']
            
            self.criterion = nn.CrossEntropyLoss().to(self.args['device'])

            self.threshold = None

    #主要运行函数
    def train(self,trainloader, testloader,per_cls_weights):
        ######## 训练部分 ########
        best_TP,best_FP,best_TN,best_FN = 0,0,0,0
        best_Accu,best_prec,best_rec = 0,0,0
        best_f1 = 0
        # filename = f".\console\{self.args['random_seed']}_{self.args['noise_rate']}_{self.args['imb_ratio']}_{self.args['train_norm_size']}_{self.args['imb_method']}.txt"
        for i in range(self.args['finally_epoch']):
            start_time = timeit.default_timer()
            ######################################################
            self.train_model(i, trainloader,per_cls_weights)
            if i%20 == 19 or i == 0:
                TP,FP,TN,FN = self.eval(testloader, self.model, i)
                rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
                prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
                Accu = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
                F1 = 2 * rec * prec / (rec + prec) if (rec + prec) != 0 else 0
                if F1>best_f1:
                    best_TP,best_FP,best_TN,best_FN = TP,FP,TN,FN
                    best_Accu,best_prec,best_rec = Accu,prec,rec
                    best_f1 = F1
                # with open(filename, 'w') as f:
                #     f.write('-' * 25 )
                #     f.write("TP:\t" + str(TP) + '\t|| ')
                #     f.write("FP:\t" + str(FP) + '\t|| ')
                #     f.write("TN:\t" + str(TN) + '\t|| ')
                #     f.write("FN:\t" + str(FN))
                #     f.write("Recall:\t{:6.4f}".format(rec)+ '\t|| ')
                #     f.write("Precision:\t{:6.4f}".format(prec))
                #     f.write("Accuracy:\t{:6.4f}".format(Accu)+'\t|| ')
                #     f.write("F1:\t{:6.4f}".format(F1))
        print("TP:\t" + str(best_TP), end='\t|| ')
        print("FP:\t" + str(best_FP), end='\t|| ')
        print("TN:\t" + str(best_TN), end='\t|| ')
        print("FN:\t" + str(best_FN))
        print("Recall:\t{:6.4f}".format(best_rec), end='\t|| ')
        print("Precision:\t{:6.4f}".format(best_prec))
        print("Accuracy:\t{:6.4f}".format(best_Accu), end='\t|| ')
        print("F1:\t{:6.4f}".format(best_f1))
        # with open(filename, 'w') as f:
        #     f.write('-' * 25 )
        #     f.write("TP:\t" + str(best_TP)+ '\t|| ')
        #     f.write("FP:\t" + str(best_FP)+ '\t|| ')
        #     f.write("TN:\t" + str(best_TN)+ '\t|| ')
        #     f.write("FN:\t" + str(best_FN))
        #     f.write("Recall:\t{:6.4f}".format(best_rec)+ '\t|| ')
        #     f.write("Precision:\t{:6.4f}".format(best_prec))
        #     f.write("Accuracy:\t{:6.4f}".format(best_Accu)+ '\t|| ')
        #     f.write("F1:\t{:6.4f}".format(best_f1))
        #     f.write("Time:\t" + str(timeit.default_timer() - start_time))
        
    
            
                                                                    

    def train_model(self, epoch, trainloader,per_cls_weights):
        self.model.train()
        batch_idx = 0
        losses = AverageMeter()
        criterion = FocalLoss(gamma=2, weight=None)
        for i, (x, y,_) in enumerate(trainloader):
            x = x.to(self.args['device'])
            y = y.to(self.args['device'])
            logits = self.model(x)
            y = y.long()
            self.optimizer.zero_grad()
            # loss = nn.CrossEntropyLoss()(logits, y)
            # loss = F.cross_entropy(logits, y, weight=per_cls_weights, reduction='mean')
            loss = criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), len(logits))

            batch_idx += 1
        print('Epoch [%3d/%3d] Loss: %.2f' % (epoch, self.args['finally_epoch'], losses.avg))

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

        TP,FP,TN,FN = evaluate_true_pred_label(model_true, model_preds)
        cm = confusion_matrix(model_true, model_preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()

        acc = 100 * float(correct) / float(total)
        # print(class_acc)
        print('Epoch [%3d/%3d] Test Acc: %.2f%%' %(epoch, self.args['finally_epoch'], acc))
        gap_cls = int(math.ceil(self.args['num_class'] / 2))

        return TP,FP,TN,FN