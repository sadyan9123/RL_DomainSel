# encoding: utf-8
# !/usr/bin/env python
import os
import json
import random
import numpy as np
from collections import deque
from RL.ForwardNet import ForwardNet3L, ForwardNet5L, ForwardNet5LReLU,ForwardNet3LConfigable, ForwardNet5LSigmoid
from RL.QLeaner import QLeaner
import math
import matplotlib.pyplot as plt
import time
import argparse
from PreProc import safe_log
#from PreProc import proc_reward
from PreProc import proc_feature_domainsel, proc_feature_for_train
#from PreProc import proc_feature_file,feature_cat,reward2name,reward_strategy_default,reward_strategy_avoid_timeout,reward_strategy_seek_succ,reward_strategy_avoid_timeout_p15,reward_strategy_avoid_timeout_p20,reward_strategy_avoid_timeout_p50,reward_strategy_avoid_timeout_p100
from PreProc import proc_feature_file,feature_cat,reward2name

# 修改train.py里的训练文件夹地址，也就是刚才生成的domain文件夹
# 把DEBUG_TRAINING_SET和REPORT_REDUNDENT置true
# 执行命令：python train.py  --nn_arch ForwardNet3LConfigable --reward_design a-0.1_s5_e0_b-2_p-15 --nn_extra h512_aSigmoid --gamma 0.9 --rich_feature True --epoch 10000000 --batch_size 2048
    #gamma是看的远的概率
    #可选的网络类型是
            # name2nn = {'ForwardNet3L':ForwardNet3L,
            #    'ForwardNet5L':ForwardNet5L,
            #    'ForwardNet5LReLU':ForwardNet5LReLU,
            #    'ForwardNet3LConfigable':ForwardNet3LConfigable,
            #    'ForwardNet5LSigmoid':ForwardNet5LSigmoid}
    #reward类型是：
            # reward_remap = {'just_alive': a,
            #         'succ_done': s,
            #         'even': e,
            #         'bad_choice': b,
            #         'timeout': p}
    
# 将输出的大于5000的项目复制粘贴，（Pre文件和Train里都粘贴）
# 把DEBUG_TRAINING_SET和REPORT_REDUNDENT置fasle
# 再根据domain训练神经网络：
    #执行命令：python train.py --nn_arch ForwardNet3LConfigable --reward_design a-0.1_s5_e0_b-2_p-15 --nn_extra h512_aSigmoid
# 可能会出现维度不匹配，在TDD里改

TRAINING_SET_DIR = '../traindata2'  # 训练集文件夹的地址
DEBUG_TRAINING_SET = False  # 对训练集进行debug
SHOW_PLOT = False
REPORT_REDUNDENT = False     # 是否报告冗余项
FEATURE_FILE_DIM = 25  
FEATURE_DOMAIN_SEL_DIM_EXT = 1  
FEATURE_DOMAIN_SEL_DIM_ORG = 72
FEATURE_DOMAIN_SEL_DIM_ALL = FEATURE_DOMAIN_SEL_DIM_ORG + FEATURE_DOMAIN_SEL_DIM_EXT  # 全部feature domainsel的维度
FEATURE_DIM = FEATURE_FILE_DIM + FEATURE_DOMAIN_SEL_DIM_ALL  # 全部feature的维度
REWARD_FUNC = None
NN_PARA_DIR = None
IS_FULL_FEATURE = False

idx_more_than_5k_feature_domainsel = [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,39,40,41,42,46,47,48,51,52,53,55,56,57,58,59,60,64,65,66,67]
idx_more_than_5k_feature_file = [9,14,20,22,23,24]

# 对训练集进行debug的代码
if DEBUG_TRAINING_SET:
    
    FEATURE_FILE_NAME = open('FEATURE_STATIC_25.txt').readlines()
    FEATURE_DOMAINSEL_NAME = open('FEATURE_VA-PA_72.txt').readlines()
    FEATURE_FILE_NAME = [s.strip() for s in FEATURE_FILE_NAME]
    FEATURE_DOMAINSEL_NAME = [s.strip() for s in FEATURE_DOMAINSEL_NAME]
    print(len(FEATURE_DOMAINSEL_NAME))
    print(len(set(FEATURE_DOMAINSEL_NAME)))
    DEBUG_ALL_FG_SET = list()
    DEBUG_ALL_FF_SET = list()
    assert len(FEATURE_FILE_NAME) == FEATURE_FILE_DIM
    assert len(FEATURE_DOMAINSEL_NAME) == FEATURE_DOMAIN_SEL_DIM_ORG


    def feature_domainsel_debug():
        return feature_debug(DEBUG_ALL_FG_SET, FEATURE_DOMAINSEL_NAME, 'feature domainsel')


    def feature_file_debug():
        return feature_debug(DEBUG_ALL_FF_SET, FEATURE_FILE_NAME, 'feature file')


    def feature_debug(dataset, name_list, descript):
        assert len(dataset) > 0
        print('开始对 {} 进行 debug 共{}个待测项'.format(descript, len(dataset)))

        report = list()
        redundent_ids = list()
        DIM = len(dataset[0])

        values = list()

        for i in range(DIM):
            all_v = [f[i] for f in dataset]
            values.append(all_v)

            if REPORT_REDUNDENT and len(set(all_v)) == 1:
                report.append('监测到冗余项 f[{}] {} # {}'.format(i, all_v[0], name_list[i]))
                redundent_ids.append(i)


        for r in report:
            print(r)
        print('冗余项共计 {} 个'.format(len(report)))
        print(redundent_ids)

        print('{} 非冗余项大于5000的如下：'.format(descript))
        for i in range(DIM):

            _all = values[i]
            _min = min(_all)
            _max = max(_all)
            kind = len(set(_all))
            if _min == _max:
                assert 1 == kind
                continue
            # if _max<=100:
            #     continue

            show = list(set(_all))
            show.sort()
            show.reverse()

            mean = np.mean(show)
            std = np.mean(show)


            #print('id:{} {:.0f} -> {:.0f}  {}种 均值{:.1f} 标准差{:.1f} # {} \t{}'.format(i,_min, _max, kind,mean, std, name_list[i], show))
            if _max>5000:
                print(i, end=',')


            if SHOW_PLOT:
                show.reverse()
                x = range(len(show))
                plt.plot(x, show)
                plt.show()

# 样本的计数器，分别代表未进入cegar的样本、只走了一步的样本、和未结束的样本
counter_no_cegar = 0
counter_1_cegar = 0
counter_not_end = 0  # 未走完的样本
counter_crashed = 0  # 未走完的样本中，最后一项不是reward（算reward时死掉的样本）


# 把int型的action改成vector [0,1][1,0]
def proc_act(act):
    if act == -1:
        return None
    ret = np.zeros(2)
    assert act in {0, 1}, act
    ret[act] = 1
    return ret


# 把单个json seq文件加载成dict seq，并简单清理，处理一下双reward，补一下time-out，把reward由数值换成枚举
def file2json(fname):
    processed_feature_list = list()

    with open(fname) as f:
        #print(fname)
        l1 = json.loads(f.readline())
        l2 = json.loads(f.readline())
        
        #print(l1['type'] )
        assert l1['type'] == 'feature_file'
        assert l2['type'] == 'ff_ack'

        # 文件级feature
        feature_file = l1['f']
        assert len(feature_file) == FEATURE_FILE_DIM
        if DEBUG_TRAINING_SET: DEBUG_ALL_FF_SET.append(feature_file.copy())
        #feature_file = proc_feature_file(feature_file)# TODO @杨璨 hello


        # 不需要解析的消息类型,是RL端加上的标记
        filter_list = {'ff_ack', 'r_ack', 'action_domainsel'}

        for l in f.readlines():

            dict_obj = json.loads(l)
            if dict_obj['type'] in filter_list:
                continue

            # 删除NN用不到的项
            dict_obj.pop('id')
            dict_obj.pop('type')
            if 'factors' in dict_obj:
                dict_obj.pop('factors')

            # 对reward、feature、action进行标准化
            if 'r' in dict_obj:
                dict_obj['r'] = reward2name(dict_obj['r'])
            if 'f' in dict_obj:
                f=dict_obj['f']
                if DEBUG_TRAINING_SET:
                    DEBUG_ALL_FG_SET.append(f.copy())
                #dict_obj['f'] = proc_feature_domainsel(f)# TODO @杨璨 hello
            if 'used_action' in dict_obj:#把int变成[1,0]或[0,1]
                dict_obj['used_action'] = proc_act(dict_obj['used_action'])

            # 拼接
            processed_feature_list.append(dict_obj)




    msg_len = len(processed_feature_list)

     # 处理一下过短样本
    if msg_len <= 2:
        #print(fname)
        if msg_len == 0:
            global counter_no_cegar # 未进入ceagr
            counter_no_cegar += 1

        if msg_len == 2:
            global counter_1_cegar  # 只走了一步cegar
            counter_1_cegar += 1
        
        assert msg_len >= 0 and msg_len != 1
        return None, None

    # 这里清洗一下结尾的双reward
    last = processed_feature_list[-1]
    second_last = processed_feature_list[-2]

    is_full_sampe = ('r' in last) and ('r' in second_last)

    # 如果结尾不是双reward，证明不是完整样本
    if not is_full_sampe:
        # print('放弃不完整样本: ' + fname)
        global counter_not_end
        counter_not_end += 1

        if 'used_action' not in last:
            global counter_crashed
            counter_crashed+=1
            return None, None


        # TODO 这里应该人工加一行time out（需要考虑结尾对齐）
        assert 'used_action' in last
        last['r'] = 'timeout'
        # for l in processed_feature_list:
        #     print(l)
        # print('end')
        return processed_feature_list, feature_file
        #return None, None

    # 将最后一个action（倒数第二行），保存在最后一行
    last['used_action'] = second_last['used_action']

    # 删掉倒数第二行（重复的reward）
    del processed_feature_list[msg_len - 2]



    return processed_feature_list, feature_file


# terminal可能有错，这里强行重写terminal。最后一行为1，其余为0
def reset_terminal(feature_list):
    for l in feature_list:
        if 'terminal' in l: l['terminal'] = 0
    feature_list[-1]['terminal'] = 1    # 最后一个元素


def assign_reward(feature_list):
    global REWARD_FUNC
    for l in feature_list:
        if 'r' in l:
            #l['r']=reward_strategy_default(l['r'])
            l['r'] = REWARD_FUNC(l['r'])



# 强化学习的原料是变迁对  trans = (s_t, a_t, r_t, s_t1, terminal)
# 该函数把一个文件，拆成若干变迁对
def file2tran_seq(clean_list, ff):
    #clean_list, ff = file2json(fn)


    if clean_list is None:  # 该文件有错
        return None

    reset_terminal(clean_list)  # 重置terminal
    assign_reward(clean_list)   # 赋值reward

    #if clean_list[-1]['r']!=10:
    #print(clean_list[-1])

    N = len(clean_list)
    assert N % 2 == 0

    ret = list()
    i = 0
    while i != (N):
        s_t = clean_list[i]['f']
        
        a_t = clean_list[i + 1]['used_action']
        r_t = clean_list[i + 1]['r']
        terminal = clean_list[i + 1]['terminal']

        if terminal != 1:
            s_t1 = clean_list[i + 2]['f']
        else:
            s_t1 = [0 for _ in range(len(s_t))]# 由于NN架构的需求，这边需要padding一个假的s_t1

        # 添加feature
        s_t = feature_cat(ff, s_t)
        s_t1 = feature_cat(ff, s_t1)
        #print("this length of one state feature is {}".format(len(s_t)))

        # np化
        s_t = np.array(s_t)
        s_t1 = np.array(s_t1)

        # 添加维度
        s_t = np.stack((s_t,), axis=1)  # 现在只是单纯升一维，以后如有需求，可以做成多时间戳
        s_t1 = np.stack((s_t1,), axis=1)
        trans = (s_t, a_t, r_t, s_t1, terminal)

        # assert s_t.shape[0] == FEATURE_DIM, s_t.shape[0]
        # assert s_t1.shape[0] == FEATURE_DIM, s_t1.shape[0]
        ret.append(trans)

        i += 2
    return ret


# 把整个文件夹都拆成seq序列
def dir2_trans_seq(train_dir):
    train_dir += '/'
    assert os.path.exists(train_dir), train_dir

    ret = list()
    messages, ffs = [], []
    for dirpath, dirnames, filenames in os.walk(train_dir):
        print('正在搜索子文件夹：'+dirpath)
        for filepath in filenames:
            fn = os.path.join(dirpath, filepath)
            #trans_seq = file2tran_seq(fn)
            # if os.path.splitext(fn)[1] != '.domain':
            #     print('发现非训练数据：'+fn)
            #     continue
            clean_list, ff = file2json(fn)
            if clean_list:
                messages.append(clean_list)
                ffs.append(ff)
            # if trans_seq is not None:
            #     ret.extend(trans_seq)
    global counter_not_end, counter_1_cegar, counter_no_cegar
    print('未进入cegar的样本：' + str(counter_no_cegar))
    print('只走了1步的样本：' + str(counter_1_cegar))
    print('未结束的样本：' + str(counter_not_end))
    print('Crash的样本：' + str(counter_crashed))
    proc_feature_for_train(messages, ffs, IS_FULL_FEATURE)

    for ms, ff in zip(messages, ffs):
        trans_seq = file2tran_seq(ms, ff)
        ret.extend(trans_seq)
    return ret






# 把得到 trans seq 喂给强化学习,如果要报告冗余项就直接退出了
def train(trans_seq, nn_arch,gamma, epoch,nn_extra,batch_size):
    if DEBUG_TRAINING_SET:
        feature_domainsel_debug()
        feature_file_debug()
        exit()

    ACTIONS_NUM = 2  # number of valid actions
    FEATURE_DIM = len(trans_seq[0][0])
    
    StateShape = [None, FEATURE_DIM, 1]
    print("feature 长度" + str(FEATURE_DIM))


    name2nn = {'ForwardNet3L':ForwardNet3L,
               'ForwardNet5L':ForwardNet5L,
               'ForwardNet5LReLU':ForwardNet5LReLU,
               'ForwardNet3LConfigable':ForwardNet3LConfigable,
               'ForwardNet5LSigmoid':ForwardNet5LSigmoid}
    NN = name2nn[nn_arch]
    print("yyyyyyyyyyyyyyyyyyyyyyyy")
    print(StateShape)
    net = NN(StateShape, ACTIONS_NUM, nn_extra)
    global NN_PARA_DIR
    assert NN_PARA_DIR is not None
    ql = QLeaner(net, NN_PARA_DIR, gamma, epoch, batch_size)

    print('变迁数量：' + str(len(trans_seq)))
    # exit()

    for trans in trans_seq:
        s_t, a_t, r_t, s_t1, terminal = trans
        ql.add_trans(s_t, a_t, r_t, s_t1, terminal)

    while True:
        ql.sgd()



def generatePandasData(trans_seq):
    print(trans_seq[0][0])
    print(trans_seq[0][0].shape[0])
    dim = trans_seq[0][0].shape[0]
    data = np.zeros(shape=(len(trans_seq), dim))
    # print(trans_seq[0])
    for i, row in enumerate(trans_seq):
        # data = np.vstack(data, row[0])
        data[i, :] = row[0]
    print(data.shape)
    np.savetxt('test.out', data, delimiter=',')

def main():
    parser = argparse.ArgumentParser(description='训练一个RL模型')
    #parser.add_argument('--reward_strategy', dest='reward_strategy', required=True, help='使用的reward的策略')
    parser.add_argument('--reward_design', dest='reward_design', required=True, help='使用的reward的具体数值')
    # 例如
    # python train.py --rich_feature True --epoch 10000 --gamma 0.9 --reward_design a-0.1_s5_e0_b-2_p-15 --nn_arch ForwardNet3LConfigable --nn_extra h512_aNone --batch_size 2048

    parser.add_argument('--nn_arch', dest='nn_arch', required=True, help='使用的NN的架构')
    parser.add_argument('--nn_extra', dest='nn_extra', required=True, help='NN的额外信息')
    parser.add_argument('--rich_feature', dest='is_full_feature', default=False, help='是否进行全数据升维')

    parser.add_argument('--gamma', dest='gamma', required=True, help='强化学习gamma值')
    parser.add_argument('--epoch', dest='epoch', required=True, help='训练多少轮后停止')
    parser.add_argument('--batch_size', dest='batch_size', required=True, help='batch_size, 开始的实验都在用128 * 16')


    args = parser.parse_args()


    a,s,e,b,p=args.reward_design.split('_')
    assert a[0]=='a' and s[0] =='s' and e[0]=='e' and b[0]=='b' and p[0]=='p'
    a,s,e,b,p = float(a[1:]),float(s[1:]),float(e[1:]),float(b[1:]),float(p[1:])
    assert a<0 and s>0 and e==0 and b<0 and p<0
    reward_remap = {'just_alive': a,
                    'succ_done': s,
                    'even': e,
                    'bad_choice': b,
                    'timeout': p}

    print('reward 设计如下')
    for k in reward_remap:
        print('\t'+k+': '+str(reward_remap[k]))


    def reward_strategy_configable(r):
        assert r in reward_remap, r
        ret = reward_remap[r]
        #print('debug '+r+' -> '+str(ret))
        return ret


    # name2func = {'reward_strategy_default':reward_strategy_default,
    #              'reward_strategy_avoid_timeout':reward_strategy_avoid_timeout,
    #              'reward_strategy_seek_succ':reward_strategy_seek_succ,
    #              'reward_strategy_avoid_timeout_p15':reward_strategy_avoid_timeout_p15,
    #              'reward_strategy_avoid_timeout_p20': reward_strategy_avoid_timeout_p20,
    #              'reward_strategy_avoid_timeout_p50': reward_strategy_avoid_timeout_p50,
    #              'reward_strategy_avoid_timeout_p100': reward_strategy_avoid_timeout_p100}
    # global REWARD_FUNC
    # REWARD_FUNC=name2func[args.reward_strategy]
    global REWARD_FUNC
    REWARD_FUNC = reward_strategy_configable

    global NN_PARA_DIR
    server_start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    NN_PARA_DIR = 'NNPara'+server_start_time+'_'+args.nn_arch+'_'+args.reward_design+'_'+args.is_full_feature+'_'+args.gamma+'_'+args.epoch+'_'+args.batch_size+'_'+args.nn_extra

    global  IS_FULL_FEATURE
    IS_FULL_FEATURE = args.is_full_feature

    trans_seq = dir2_trans_seq(TRAINING_SET_DIR)
    train(trans_seq, args.nn_arch, args.gamma, args.epoch,args.nn_extra,args.batch_size)

main()

'''
ssh_m/2
cd ./exp_v3_client/remote_3
source activate py36rl
cd RL_DomainSel

cat ../exp_v3_client/remote_3/*0411*0.0*

cd ./exp_v3_server
source activate py36rl

watch -n 1 ls NN*
watch -n 1 tail 20190410.log

watch -n 1 'cat 20190419client.txt | grep start | wc -l'

watch -n 10 tail 20190419client.txt

mv 20190419client.txt 20190410晚client_gamma0.7.log

mv results results_gamma0.7_m1

tar -czvf results_gamma0.7_m1.tar.gz results_gamma0.7_m1
tar -czvf results_gamma0.7_m2.tar.gz results_gamma0.7_m2
tar -czvf results_gamma0.7_m3.tar.gz results_gamma0.7_m3

tar -czvf results_gamma0.7_m4.tar.gz results_gamma0.7_m4
tar -czvf results_gamma0.7_m5.tar.gz results_gamma0.7_m5
tar -czvf results_gamma0.7_m6.tar.gz results_gamma0.7_m6

tar -czvf results_gamma0.7_m7.tar.gz results_gamma0.7_m7
tar -czvf results_gamma0.7_m8.tar.gz results_gamma0.7_m8
tar -czvf results_gamma0.7_m9.tar.gz results_gamma0.7_m9


在tmux中，以下实验，按照从左往右，从第一行到第三行到顺序进行
source activate py36rl

python TDD_Server_mt.py --NN_dir NNPara* --port 5000 --rich_feature True
echo {\"ip\": \"127.0.0.1\",\"port\": 5000} > ./RLServerInfo.config
benchexec -N 6 -c 1 cpa-refsel.xml > 20190419client.txt




2019年4月10日晚，实验内容，gamma 0.7 下的 nn-reward全覆盖实验



python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_default --nn_arch ForwardNet3L > 20190410.log
python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_default --nn_arch ForwardNet5L > 20190410.log
python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_default --nn_arch ForwardNet5LReLU > 20190410.log

python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_avoid_timeout --nn_arch ForwardNet3L > 20190410.log
python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_avoid_timeout --nn_arch ForwardNet5L > 20190410.log
python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_avoid_timeout --nn_arch ForwardNet5LReLU > 20190410.log

python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_seek_succ --nn_arch ForwardNet3L > 20190410.log
python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_seek_succ --nn_arch ForwardNet5L > 20190410.log
python train.py --rich_feature True --epoch 10000 --gamma 0.7 --reward_strategy reward_strategy_seek_succ --nn_arch ForwardNet5LReLU > 20190410.log


watch -n 10 ls NNPara*ForwardNet3L*reward_strategy_default*
watch -n 10 ls NNPara*ForwardNet5L_*reward_strategy_default*
watch -n 10 ls NNPara*ForwardNet5LReLU*reward_strategy_default*

watch -n 10 ls NNPara*ForwardNet3L*reward_strategy_avoid_timeout*
watch -n 10 ls NNPara*ForwardNet5L_*reward_strategy_avoid_timeout*
watch -n 10 ls NNPara*ForwardNet5LReLU*reward_strategy_avoid_timeout*

watch -n 10 ls NNPara*ForwardNet3L*reward_strategy_seek_succ*
watch -n 10 ls NNPara*ForwardNet5L_*reward_strategy_seek_succ*
watch -n 10 ls NNPara*ForwardNet5LReLU*reward_strategy_seek_succ*

cd exp_v3_server
ls ./train_M*
bash ./train_M*

watch -n 10 ls NN*0411-09*

benchexec -N 6 -c 1 cpa-refsel.xml > baseline_cpu6.log

scp ./train_M1.sh $M1:/root/exp_v3_server
scp ./train_M2.sh $M2:/root/exp_v3_server
scp ./train_M3.sh $M3:/root/exp_v3_server

scp ./train_M4.sh $M4:/root/exp_v3_server
scp ./train_M5.sh $M5:/root/exp_v3_server
scp ./train_M6.sh $M6:/root/exp_v3_server

scp ./train_M7.sh $M7:/root/exp_v3_server
scp ./train_M8.sh $M8:/root/exp_v3_server
scp ./train_M9.sh $M9:/root/exp_v3_server


scp ./auto_test_4nn.sh $M1:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M2:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M3:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M4:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M5:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M6:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M7:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M8:/root/exp_v3_client/remote_3
scp ./auto_test_4nn.sh $M9:/root/exp_v3_client/remote_3


scp  $M7:/root/exp_v3_client/remote_3/results_gamma0.0_m7.tar.gz .
scp  $M7:/root/exp_v3_client/remote_3/results_gamma0.3_m7.tar.gz .
scp  $M7:/root/exp_v3_client/remote_3/results_gamma0.5_m7.tar.gz .
scp  $M7:/root/exp_v3_client/remote_3/results_gamma0.9_m7.tar.gz .


scp  $M8:/root/exp_v3_client/remote_3/results_gamma0.0_m8.tar.gz .
scp  $M8:/root/exp_v3_client/remote_3/results_gamma0.3_m8.tar.gz .
scp  $M8:/root/exp_v3_client/remote_3/results_gamma0.5_m8.tar.gz .
scp  $M8:/root/exp_v3_client/remote_3/results_gamma0.9_m8.tar.gz .


scp  $M9:/root/exp_v3_client/remote_3/results_gamma0.0_m9.tar.gz .
scp  $M9:/root/exp_v3_client/remote_3/results_gamma0.3_m9.tar.gz .
scp  $M9:/root/exp_v3_client/remote_3/results_gamma0.5_m9.tar.gz .
scp  $M9:/root/exp_v3_client/remote_3/results_gamma0.9_m9.tar.gz .

ls






'''


#
