import numpy as np
from sklearn import metrics as metrics

def set_random_seed(seed=42, deterministic=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass

def evaluate_true_pred_label(y_true, y_pred, desc='', para='strong'):

    '''根据true和pred评估结果，true在前'''
    try:
        num = y_true.shape[0]
    except:
        num = len(y_true)
    if num == 0:
        return

    print('-' * 10 + desc + '-' * 10)
    cf_flow = metrics.confusion_matrix(y_true, y_pred)
    if len(cf_flow.ravel()) == 1:
        if y_true[0] == 0:
            TN, FP, FN, TP = cf_flow[0][0], 0, 0, 0
        elif y_true[0] == 1:
            TN, FP, FN, TP = 0, 0, 0, cf_flow[0][0]
        else:
            raise Exception("label error")
    else:
        TN, FP, FN, TP = cf_flow.ravel()

    rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
    Accu = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0

    F1 = 2 * rec * prec / (rec + prec) if (rec + prec) != 0 else 0
    if para.lower() == 'strong'.lower():
        print("TP:\t" + str(TP), end='\t|| ')
        print("FP:\t" + str(FP), end='\t|| ')
        print("TN:\t" + str(TN), end='\t|| ')
        print("FN:\t" + str(FN))
        print("Recall:\t{:6.4f}".format(rec), end='\t|| ')
        print("Precision:\t{:6.4f}".format(prec))
        print("Accuracy:\t{:6.4f}".format(Accu), end='\t|| ')
        print("F1:\t{:6.4f}".format(F1))
    else:
        print("\tTP \t" + str(TP), end='\t - ')
        print("\tFP \t" + str(FP), end='\t - ')
        print("\tTN \t" + str(TN), end='\t - ')
        print("\tFN \t" + str(FN))
        print("\tRecall \t{:6.4f}".format(rec), end='\t - ')
        print("\tPrecision \t{:6.4f}".format(prec))
        print("\tAccuracy \t{:6.4f}".format(Accu), end='\t - ')
        print("\tF1 \t{:6.4f}".format(F1))
    return TP,FP,TN,FN

def get_overlap(kmeans_labels, dbscan_labels):
    """
    This function returns the overlapping clusters between two sets of clusters.
    """
    clf_clusters1 = []
    clf_clusters2 = []

    for i in range(len(np.unique(kmeans_labels))):
        clf_clusters1.append([])
    for i_idx in range(len(kmeans_labels)):
        clf_clusters1[kmeans_labels[i_idx]].append(i_idx)

    for i in range(len(np.unique(dbscan_labels))):
        clf_clusters2.append([])
    for i_idx in range(len(dbscan_labels)):
        clf_clusters2[dbscan_labels[i_idx]].append(i_idx)

    my_clusters = []
    for set1 in clf_clusters1:
        for set2 in clf_clusters2:
            new_set = []
            for i_idx in set1:
                if i_idx in set2:
                    new_set.append(i_idx)
            if len(new_set) >= 1:
                # print(len(set1), len(set2))
                my_clusters.append(new_set)
    return np.array(my_clusters)