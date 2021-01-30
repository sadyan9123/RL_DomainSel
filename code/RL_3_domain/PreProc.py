# encoding: utf-8
# !/usr/bin/env python
import os
import json
import random
import numpy as np
from collections import deque
from RL.QLeaner import QLeaner
import math
import matplotlib.pyplot as plt
from utils.pdProcess import *

# TODO 这段代码是 train.py 的直接拷贝版本，耦合度很高。是方便法，以后需要去优化
#idx_more_than_5k_feature_domainsel = [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,40,41,42,43,49,50,51,52,53,54,58,61,62,63,64]
#idx_more_than_5k_feature_file = [8,9,14,20,22,23,24]

idx_more_than_5k_feature_domainsel = [0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,22,23,25,26,27,35,36,37,38,43,44,46,47,48,49,50,51,52,53,54,55,56,60,61,62,63,68,69,71,72,73,85,86,87,88]
idx_more_than_5k_feature_file = [9,10,12,14,20,22,23,24]

# 安全的log函数
def safe_log(x):
    assert x >= 0
    if x == 0: return 0
    ret = math.log(x)

    return int(ret) # TODO 为了统计方便，加了int。记得取消


def reward2name(r):
    r2n = {1:'just_alive',
           100:'succ_done',
           0:'even',
           -1:'bad_choice',
           -7:'bad_choice',
           -9:'bad_choice',
           -100:'timeout'}
    return r2n[r]

# TODO feature domainsel的预处理
def proc_feature_domainsel(f):

    # 对于最大值大于5k的属性，直接加log
    for i in range(len(f)):
        if i in idx_more_than_5k_feature_domainsel:
            f[i]=safe_log(f[i])

    #if DEBUG_TRAINING_SET: DEBUG_ALL_FG_SET.append(f.copy())
    # return [7,]
    # return np.array([7.0,])
    # print(f)

    if f[0] == 2147483647:
        f.append(1)
        f[0] = 0
    else:
        f.append(0)

    # print(f[49])
    # assert f[49] == 633559  # 第49个feature数值恒等于633559
    # 注：有时可能为185789
    f[49] = 0

    # print(max(f))

    #assert len(f) == FEATURE_DOMAIN_SEL_DIM_ALL, len(f)

    #return [safe_log(x) for x in f]
    return f
    # return np.array(f)


# TODO feature file的预处理
def proc_feature_file(f):
    # 对于最大值大于5k的属性，直接加log
    for i in range(len(f)):
        if i in idx_more_than_5k_feature_file:
            f[i]=safe_log(f[i])

    #if DEBUG_TRAINING_SET: DEBUG_ALL_FF_SET.append(f.copy())
    return f

# 把feature file和feature domainsel拼起来，需要深拷贝
def feature_cat(ff, s_t):
    ret = ff.copy()
    ret.extend(s_t)
    return ret



def proc_feature_for_test(f, ff, is_full_feature):
    full_feature = ff.copy()
    if f[0] == 2147483647:
        f.append(1)
        f[0] = 0
    else:
        f.append(0)
    full_feature.extend(f)
    return getFeaturedSingleTestData(full_feature, is_full_feature)


def proc_feature_for_train(messages, ffs, is_full_feature):
    data = []
    for ms, ff in zip(messages, ffs):
        for m in ms:
            if 'f' in m:
                if m['f'][0] == 2147483647:
                    m['f'].append(1)
                    m['f'][0] = 0
                else:
                    m['f'].append(0)
                fs = ff.copy()
                fs.extend(m['f'])
                data.append(fs)
    pdata, ff_length = getFeaturedTrainData(data, is_full_feature)
    print(len(pdata), len(pdata[0]))
    #ff_length = len(ffs[0])
    i = 0
    for idx, ms in enumerate(messages):
        for m in ms:
            if 'f' in m:
                #there can't gurantte the original order
                m['f'] = pdata[i][ff_length:]
                ffs[idx] = pdata[i][:ff_length]
                i += 1





# def proc_feature_for_train(feature_list, file_feature):
#     data = []
#     indexs = []
#     for row in feature_list:
#         indexs.append(0)
#         if 'f' in row:
#             indexs[-1] = 1
#             if row['f'][0] == 2147483647:
#                 row['f'].append(1)
#                 row['f'][0] = 0
#             else:
#                 row['f'].append(0)
#             fs = file_feature.copy()
#             fs.extend(row['f'])
#             data.append(fs)
#     data = getFeaturedTrainData(data)
#     it = 0
#     for row, should_update in zip(feature_list, indexs):
#         if should_update == 1:
#             row['f'] = data[it]
#             it += 1




