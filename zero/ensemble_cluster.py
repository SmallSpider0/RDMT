import numpy as np
import scipy.stats
# import scipy
import torch
# from scipy import sparse
# import faiss
import torch.optim as optim
# from faiss import normalize_L2
import torch.nn.functional as F
from scipy.sparse import spdiags
from sklearn.metrics import silhouette_samples
import sklearn.metrics as metrics
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from my_tool import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Ensemble_Cluster(
                    X,
                    y,
                    score=None,
                    PCA_parameter=0.99,
                    clf_name='KMeans',
                    # clf_name='GaussianMixture',
                    clf_max_cluster_num=20,
                    ):
        Cluster_label_list=[]
        #######################################################
        # Step0:    通过PCA对高维数据降维
        #######################################################
        PCA_flag = True
        if PCA_flag:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=PCA_parameter)
            # PCA_formed_X = pca.fit_transform(X)
            try:  # 此处可能出现数学域错误，需要修改pca.py文件
                PCA_formed_X = pca.fit_transform(X)
            except:
                X_update = np.array(X) + 1e-99
                PCA_formed_X = pca.fit_transform(X_update)
            print('[-] PCA_parameter: {} , need_feature_len: {:4d}'.format(PCA_parameter, len(PCA_formed_X[0])))
        else:
            PCA_formed_X = X

        ########################################################
        # Step1:  kmeans聚类
        ########################################################
        Trying_Clusters_List = np.arange(clf_max_cluster_num // 2, clf_max_cluster_num + 1)
        # bestK, best_sih = 0, 0
        bestK, best_sih = 0, -1
        for i_cluster in tqdm(Trying_Clusters_List, desc='clu1'):
            if clf_name.lower() == 'KMeans'.lower():
                from sklearn.cluster import KMeans

                model = KMeans(n_clusters=i_cluster, n_init=10)
            elif clf_name.lower() == 'GaussianMixture'.lower() or clf_name.lower() == 'GMM'.lower():
                from sklearn.mixture import GaussianMixture

                model = GaussianMixture(n_components=i_cluster, covariance_type='full', random_state=0)
            else:
                model = None

            # 模型拟合
            model.fit(PCA_formed_X)
            kmeans_labels = model.predict(PCA_formed_X)
            sih = metrics.silhouette_score(PCA_formed_X, kmeans_labels)

            if sih >= best_sih:
                if sih >= 0:
                    print('[-] new best at: clusters_num :{:4d}, sih:{}'.format(i_cluster, sih))
                best_sih = sih
                bestK = i_cluster

        # 确定kmeans参数，执行fit以及后续步骤
        print('Applying cluster_num at {}...'.format(bestK))

        if clf_name.lower() == 'KMeans'.lower():
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
            # or by the following
            # model = KMeans(n_clusters=bestK, n_init=10)
        elif clf_name.lower() == 'GaussianMixture'.lower() or clf_name.lower() == 'GMM'.lower():
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(n_components=bestK, covariance_type='full', random_state=0)
        else:
            model = None

        model.fit(PCA_formed_X)
        PCA_formed_yhat = model.predict(PCA_formed_X)

        my_Clusters = []
        for i in range(len(np.unique(kmeans_labels))):
            my_Clusters.append([])
        for i_idx in range(len(kmeans_labels)):
            my_Clusters[kmeans_labels[i_idx]].append(i_idx)

       
        print("Cluster num:{}".format(len(my_Clusters)))

        Clu_X, Clu_label, sil_set, Clu_score = [], [], [], []
        for i_idx in range(len(my_Clusters)):
            Clu_label.extend([i_idx] * len(my_Clusters[i_idx]))
            Clu_X.extend(np.array(PCA_formed_X)[my_Clusters[i_idx]])
        sih = metrics.silhouette_score(Clu_X, Clu_label, metric='euclidean') if max(Clu_label) >= 1 else 0
        sil_set = silhouette_samples(Clu_X, Clu_label, metric='euclidean')

        return Clu_X, Clu_label, sil_set, my_Clusters

##根据距离构建相似性矩阵
def construct_matrix(sil_set,
                    Clu_X,
                    Clu_label,
                    # s,
                    ):

    alpha = 0.6
    # search for the graph
    n = len(Clu_X)
    A = np.zeros((n, n))
    Clu_label = np.array(Clu_label)

    # ----构造欧氏距离矩阵----
    for i in tqdm(range(len(Clu_X)), desc='dist'):
        t_label = Clu_label[i]
        same_idx = Clu_label == t_label
        max_sil = 1/np.sqrt(np.sum((Clu_X[i] - Clu_X[0: len(Clu_X)]) ** 2, axis=1))+ 1e-50
        A[i] = np.where(same_idx, max_sil, 0)
        A[i, i] = 0  # 对角线置零,不然会使矩阵有奇异性，求逆结果为NaN

    # Create and normalize the graph
    A = A + A.T
    A[A < 0] = 1e-4
    S = A.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    An = D * A * D
    I = scipy.sparse.eye(n)
    matrix = I - alpha * An

    # inv求逆
    return np.linalg.inv(matrix)

def label_propagation(Params, X, s_for_label, my_Clusters, Clu_label, matrix,
                          # Pred_bias=0.5, CLUSTER_improve_FLAG=True,
                          y_in_clustered_order=None, reli_thres=0.6):

        print('\t' + '-' * 10 + 'Start Try_Labeling...(New Label P)' + '-' * 10)

        # label propagation
        yy = np.zeros((len(y_in_clustered_order), 2))

        # Set the values in yy based on the conditions
        for i, (y_val, s_val) in enumerate(zip(y_in_clustered_order, s_for_label)):
            if s_val == 1:  # If the sample is selected
                yy[i] = [1 - y_val, y_val]  # Set [0, 1] for label=1 (malicious) or [1, 0] for label=0 (normal)

        # # Label Propagation Calculation
        Z = matrix * yy
        row_sums = Z.sum(axis=1)
        p_labels = np.where(row_sums == 0, -1, np.argmax(Z, 1))
        
        for i_cluster, clustered_sample_idxs in enumerate(my_Clusters):
            # 计算这个聚类中逻辑顺序下的索引范围
            start_idx = sum(len(cluster) for cluster in my_Clusters[:i_cluster])
            end_idx = start_idx + len(clustered_sample_idxs)

            # 获取该聚类中被选择样本的真实标签 
            # 改成获取噪声标签
            ###############################################
            selected_labels = y_in_clustered_order[start_idx:end_idx][s_for_label[start_idx:end_idx] == 1]
            ###############################################
            # 计算多数类标签的占比
            if len(selected_labels) > 0:
                majority_label_ratio = max(np.sum(selected_labels == 0), np.sum(selected_labels == 1)) / len(
                    selected_labels)
                if majority_label_ratio < float(Params['majority_label_ratio_thres']):
                    p_labels[start_idx:end_idx] = -1
            else:
                p_labels[start_idx:end_idx] = -1
        # 在p_labels转换成p_labels_physical_order之前
        # 检查被选中样本的真实标签与伪标签，如果不一致则用真实标签复位
        selected_indices = np.where(s_for_label == 1)[0]
        for idx in selected_indices:
            if p_labels[idx] != y_in_clustered_order[idx]:
                # 用真实标签复位
                p_labels[idx] = y_in_clustered_order[idx]

        p_labels_physical_order = np.zeros_like(p_labels)

        for i_cluster, clustered_sample_idxs in enumerate(my_Clusters):
            for physical_idx, cluster_idx in zip(clustered_sample_idxs, range(len(clustered_sample_idxs))):
                # 将聚类顺序下的伪标签映射到物理顺序
                p_labels_physical_order[physical_idx] = p_labels[
                    cluster_idx + sum(len(cluster) for cluster in my_Clusters[:i_cluster])]
        labeled_X = np.array(X)
        labeled_y = np.array(p_labels_physical_order)

        if labeled_y.ndim != 1:
            labeled_y = labeled_y.flatten()

        keep_mask = np.array([True] * len(X))
        keep_mask[labeled_y == -1] = False

        keep_mask_co = np.array([True] * len(X))
        keep_mask_co[labeled_y == 1] = False
        keep_mask_co[labeled_y == -1] = False

        return labeled_X[keep_mask_co], labeled_y[keep_mask_co], keep_mask,keep_mask_co

 # # 基于聚类结果和s直接传播伪标签
def Explain_and_Label_cluster_struture(self, X, s_for_label, my_Clusters, Clu_label,
                                        Pred_bias=0.5, CLUSTER_improve_FLAG=True,
                                        _true_y_test=None, reli_thres=0.5):
    print('\t' + '-' * 10 + 'Start Try_Labeling...' + '-' * 10)

    Begin_Label_idx = 0
    pred_y = np.array([-1] * len(X))
    keep_mask = np.array([False] * len(X))
    sample_idxs_list = []
    needed_label_idx = []
    Start_idx = 0
    for i_cluster in range(len(my_Clusters)):
        clustered_sample_idxs = np.array(my_Clusters[i_cluster])
        length = len(my_Clusters[i_cluster])
        Num = 0

        for i in range(length):
            if s_for_label[i + Start_idx] == 1:
                Num += 1
                needed_label_idx.append(clustered_sample_idxs[i])

        sample_idxs_list.append([Num, clustered_sample_idxs])
        Start_idx += length
    yy_list = _true_y_test[needed_label_idx]

    for idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
        if Num <= 0:
            continue
        End_Label_idx = Begin_Label_idx + Num

        # 聚类猜测
        temp_yy = yy_list[Begin_Label_idx:End_Label_idx]
        Anom_Score = np.sum(temp_yy) / len(temp_yy)
        #######################################
        majority_label_ratio = max(np.sum(temp_yy == 0), np.sum(temp_yy == 1)) / len(temp_yy)
        reli_thres = float(0.85)
        reliablity = majority_label_ratio
        #######################################
        if reliablity < reli_thres:
            if CLUSTER_improve_FLAG:
                for i_idx in range(len(temp_yy)):
                    needed_idx_at = Begin_Label_idx + i_idx
                    pred_y[needed_label_idx[needed_idx_at]] = temp_yy[i_idx]
                    keep_mask[needed_label_idx[needed_idx_at]] = True
        else:
            my_label = 1 if Anom_Score >= Pred_bias else 0

            pred_y[clustered_sample_idxs] = my_label
            keep_mask[clustered_sample_idxs] = True

            if CLUSTER_improve_FLAG:
                revise_true_NUM = 0

                for i_idx in range(len(temp_yy)):
                    needed_idx_at = Begin_Label_idx + i_idx
                    if pred_y[needed_label_idx[needed_idx_at]] != temp_yy[i_idx]:
                        revise_true_NUM += 1
                    pred_y[needed_label_idx[needed_idx_at]] = temp_yy[i_idx]
                    keep_mask[needed_label_idx[needed_idx_at]] = True

        Begin_Label_idx = End_Label_idx

    labeled_X = np.array(X)
    labeled_y = np.array(pred_y)
    if labeled_y.ndim != 1:
            labeled_y = labeled_y.flatten()

    keep_mask = np.array([True] * len(X))
    keep_mask[labeled_y == -1] = False

    keep_mask_co = np.array([True] * len(X))
    keep_mask_co[labeled_y == 1] = False
    keep_mask_co[labeled_y == -1] = False

    return labeled_X[keep_mask_co], labeled_y[keep_mask_co], keep_mask,keep_mask_co
    return np.array(X)[keep_mask], np.array(pred_y)[keep_mask], keep_mask