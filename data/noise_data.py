import numpy as np
import math
from numpy.testing import assert_array_almost_equal

def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0, select_class=0):
    ### 用不上 ###
    # if noise_type == 'pairflip':
    #     train_noisy_labels, actual_noise_rate, P = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    #############
    if noise_type == 'sym':
        train_noisy_labels, actual_noise_rate, P = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'asym':
        train_noisy_labels, actual_noise_rate, P = noisify_multiclass_asymmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    ### 不清楚干什么的 ###
    # if 'imb' in noise_type:
    #     imb_type = noise_type.split('_')[1]
    #     imb_rate = float(noise_type.split('_')[2])
    #     train_noisy_labels, actual_noise_rate, P = noisify_imb(train_labels, noise_rate, random_state=0, nb_classes=nb_classes, \
    #         imb_type=imb_type, imb_rate=imb_rate)
    #####################
    return train_noisy_labels, actual_noise_rate, P



# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y



def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.zeros((nb_classes, nb_classes))
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes):
            P[i, i] = 1. - n
            P[0, i] = n / (nb_classes - 1)
            P[i, 0] = n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise, P

def noisify_multiclass_asymmetric(y_train, noise, random_state=None, nb_classes=10):
    """错误：
        以非对称方式翻转
    """
    P = np.zeros((nb_classes, nb_classes))
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1.
        for i in range(1, nb_classes):
            P[i, i] = 1. - n
            P[i, 0] = n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('实际噪声 %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise, P

# imb_noisify
def noisify_imb(y_train, noise, random_state=None, nb_classes=10, imb_type='step', imb_rate=0.1):
    # init the matrix
    n = noise
    P = solver(n, imb_type, imb_rate, num_class=nb_classes)
    y_train_noisy = multiclass_noisify(y_train, P=P,
                    random_state=random_state)
    
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    print('Actual noise %.2f' % actual_noise)
    y_train = y_train_noisy
    print(P)

    return y_train, actual_noise, P

def solver(noise_rate, imb_type='step', imb_ratio=0.1, max_num=1000, num_class=10):
    from scipy import optimize

    eq_num = num_class * 3
    var_num = num_class ** 2

    M = np.zeros((eq_num, var_num + 1))
    # top 10 sum to 1
    for i in range(num_class):
        st = i * num_class
        for j in range(st, st + num_class):
            M[i, j] = 1
        M[i, var_num] = 1.0

    # 10: diagonal noise_rate
    for i in range(num_class, 2 * num_class):
        j = i % num_class
        M[i, j * (num_class + 1)] = 1
        M[i, var_num] = 1 - noise_rate

    cls_num_list = []
    if imb_type == 'step':
        for i in range(int(math.ceil(num_class / 2))):
            cls_num_list.append(max_num)
        for i in range(int(num_class / 2)):
            cls_num_list.append(int(max_num * imb_ratio))
    else:
        cls_num = num_class
        for cls_idx in range(cls_num):
            num = max_num * (imb_ratio**(cls_idx / (cls_num - 1.0)))
            cls_num_list.append(int(num))

    # 10: imb top 5 1000 top 5 100 
    for i in range(2 * num_class, 3 * num_class):
        # 0-10-20 ... 90
        # 1-11-21 ... 91
        # 9-19-29 ... 99
        st = i % num_class
        for j in range(st, st+var_num, num_class):
            M[i, j] = cls_num_list[int(j / num_class)]
        M[i, var_num] = cls_num_list[int(i % num_class)]

    A_eq = M[:, :var_num]
    b_eq = M[:, var_num]

    c = np.ones(var_num)

    ans = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    assert ans['success'] == True
    p = ans['x'].reshape(num_class, num_class)
    print(p)
    assert_array_almost_equal(p.sum(axis=1), np.ones(num_class))
    return p