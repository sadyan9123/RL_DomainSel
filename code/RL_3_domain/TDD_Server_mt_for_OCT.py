# -*- coding:utf-8 -*-
# !/usr/bin/env python
import os
import random
import json
import time
#import tensorflow as tf
import sys
import argparse

from socketserver import TCPServer, BaseRequestHandler, StreamRequestHandler, ThreadingTCPServer

import traceback
import PreProc
from RL.ForwardNet import ForwardNet3L,ForwardNet5L,ForwardNet5LReLU,ForwardNet3LConfigable,ForwardNet5LSigmoid
from RL.QLeaner import QLeaner
import numpy as np

server_start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
FEATURE_FILE_DIM = 25
#FEATURE_DOMAIN_SEL_DIM = 113
# FEATURE_DOMAIN_SEL_DIM = 111
FEATURE_DOMAIN_SEL_DIM = 93
DOMAIN_NUM=3

IP = '127.0.0.1'

parser = argparse.ArgumentParser(description='搭设一个训练服务器')
parser.add_argument('--NN_dir', dest='NN_dir', required=True, help='保存神经网络参数的文件夹')
parser.add_argument('--port', dest='port', required=True, help='服务器提供数据时的端口号')
parser.add_argument('--rich_feature', dest='is_full_feature', default=False, help='是否进行全数据升维')
args = parser.parse_args()
NN_dir = args.NN_dir
is_full_feature = args.is_full_feature

if NN_dir != 'None':
    print(NN_dir)
    NN_dir=NN_dir.split('/')[-1]
    nn_arch=NN_dir.split('_')[1]
    name2nn = {'ForwardNet3L':ForwardNet3L,
               'ForwardNet5L':ForwardNet5L,
               'ForwardNet5LReLU':ForwardNet5LReLU,
               'ForwardNet3LConfigable':ForwardNet3LConfigable,
               'ForwardNet5LSigmoid':ForwardNet5LSigmoid}
    NN = name2nn[nn_arch]

    h = NN_dir.split('_')[-2]
    a = NN_dir.split('_')[-1]
    extra = h+'_'+a


    if args.is_full_feature:
        StateShape = [None, 388, 1]  # there should use config to reaad 新版训练集，OCT版
    else:
        StateShape = [None, 106, 1]  #there should use config to reaad
    print("新建一个ForwardNet")
    net = NN(StateShape, 3, extra)
    print("由这个ForwardNet构建一个ql")
    ql = QLeaner(net, NN_dir, None, None, None)


port = int(args.port)


class MyBaseRequestHandlerr(BaseRequestHandler):
    # 初始化成员变量
    def setup(self):
        self.target_fn = None
        self.msg_count = 0
        self.conn_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.response_type = None
        self.start_time = time.time()

        # 该类会自动缓存feature，以便调取NN
        self.feature_file = None
        self.last_feature_domainsel = None

    # 将消息缓存到文件
    def fprint(self, msg):
        # with open(self.target_fn, 'a') as f:
        #     f.write(msg)
        #     f.flush()  # 强行文件写入
        self.msg_count += 1

        if self.msg_count % 400 == 0:
            trainsisiton_num = int(self.msg_count / 4)
            print(str(self.client_address) + ': ' + str(trainsisiton_num) + '条变迁')

    # 发送消息，自带缓存机制
    def send(self, dict_obj):
        msg = json.dumps(dict_obj)
        msg += '\n'
        print("RRRRLLLL:"+msg)
        # conn_g.sendall(bytes(msg, encoding="utf8"))
        self.request.sendall(bytes(msg, encoding="utf8"))
        self.fprint(msg)  # 自带缓存

    # 获取消息，可能返回None
    def get(self):
        try:
            # data = conn_g.recv(102400)  # 当conn端关闭时，返回空字符串
            data = self.request.recv(102400)  # 当conn端关闭时，返回空字符串

        except ConnectionResetError as e:
            print('ConnectionResetError:', e)
            return None

        # 字符串为空，连接结束
        if not data:
            return None

        msg = str(data, encoding="utf8")
        # print('debug: ' + msg)
        # fprint(msg+'\n')

        dict_obj = json.loads(msg)
        
        #print("CPACHECKER：",dict_obj)
        if self.target_fn is None:
            fn = dict_obj['src']
            
            assert fn is not None
            assert self.target_fn is None

            dir = './dump_' + server_start_time

            # if not os.path.exists(dir):
            #     os.system('mkdir ' + dir)

            self.target_fn = dir + '/' + fn + '.' + str(self.conn_time)
	    
            sys.stdout.flush()

            assert 'f' in dict_obj
            # print('yqs17 debug: length of dict_obj[f]' + str(len(dict_obj['f'])))
            #print('yqs17 debug: dict_obj[f]' + str((dict_obj['f'])))
            assert len(dict_obj['f']) == FEATURE_FILE_DIM
            self.feature_file = dict_obj['f']


        if dict_obj['type'] == 'feature_domainsel':
            # print('yqs17 debug: length of dict_obj[f]' + str(len(dict_obj['f'])))
            # print('yqs17 debug: dict_obj[f]' + str((dict_obj['f'])))
            assert len(dict_obj['f']) == FEATURE_DOMAIN_SEL_DIM, len(dict_obj['f'])
            # 注：原版是113维，修正bug后是111维
            self.last_feature_domainsel = dict_obj['f']

        self.fprint(msg)  # 自带缓存
        return dict_obj

    def send_r_ack(self):
        dict_obj = {"type": "r_ack"}
        self.send(dict_obj)

    def send_ff_ack(self):
        dict_obj = {"type": "ff_ack"}
        self.send(dict_obj)


    def send_random_domain(self):
        domain = random.randrange(DOMAIN_NUM)
        if DOMAIN_NUM == 2:
            domain_name = 'predicate' if domain else "value"
        else :
            if domain ==0 :
                domain_name = 'value'
            elif domain == 1:
                domain_name = 'oct'
            else:
                domain_name = 'predicate'
        dict_obj = {"type": "action_domainsel", 'action': domain}
        self.send(dict_obj)

    #还没改维度
    def send_QL_domain(self):
        # f = PreProc.proc_feature_domainsel(self.last_feature_domainsel) # TODO @杨璨 hello
        # ff =PreProc.proc_feature_file(self.feature_file)# TODO @杨璨 hello
        #
        # NN_input = PreProc.feature_cat(ff,f)
        #start = clock()
        NN_input = PreProc.proc_feature_for_test(self.last_feature_domainsel, self.feature_file, is_full_feature)

        NN_input = np.stack((NN_input,), axis=1)  # 现在只是单纯升一维，以后如有需求，可以做成多时间戳
        domain = ql.pred_action(NN_input, True)
        domain = np.argmax(domain)
        domain = int(domain)
        assert domain in {0,1,2}
        if domain == 0 :
            domain_name = 'oct'
        elif domain == 1:
            domain_name = 'value'
        else:
            domain_name = 'predicate'
        dict_obj = {"type": "action_domainsel", "action_name": domain_name, 'action': domain}
        #end = clock()
        #print("计算完毕：时间为", end-start )
        self.send(dict_obj)

        # TODO 将这段向量扔进NN

    def get_message(self):
        dict_obj = self.get()

        # conn结束了
        if dict_obj is None:
            return None, None

        t = dict_obj['type']

        if t =='feature_file':
            #print("feature_file: "+str(len(dict_obj['f'])))
            assert FEATURE_FILE_DIM == len(dict_obj['f']), len(dict_obj['f'])

        if t == 'feature_domainsel':
            #print("feature_domainsel: " + str(len(dict_obj['f'])))
            assert FEATURE_DOMAIN_SEL_DIM == len(dict_obj['f']), len(dict_obj['f'])
        f_num = None

        return t, f_num

    def handle(self):
        # print(self.target_fn)

        # 循环监听（读取）来自客户端的数据
        # 这句话不用打印
        print("Connected by (%r)" % (self.client_address,))
        #sys.stdout.flush()

        while True:
            try:

                T, f_num = self.get_message()

                if T is None:
                    cost_time = time.time() - self.start_time
                    cost_time = int(cost_time)
                    print("%r colsed. total time %rs." % (self.target_fn, cost_time))
                    sys.stdout.flush()
                    break

                if T == 'feature_domainsel':
                    if NN_dir == 'None':
                        self.send_random_domain()
                    else:
                        self.send_QL_domain()
                elif T == 'reward_domainsel':
                    self.send_r_ack()
                elif T == 'feature_file':
                    self.send_ff_ack()
                else:
                    assert False, T

            except:

                traceback.print_exc()

                break


if __name__ == "__main__":

    addr = (IP, port)

    # 购置TCPServer对象，

    # server = TCPServer(addr, MyBaseRequestHandlerr)            # 单线程
    server = ThreadingTCPServer(addr, MyBaseRequestHandlerr)  # 多线程

    print('服务器正在监听')
    sys.stdout.flush()
    print('本机地址：'+str(addr))
    sys.stdout.flush()
    # 启动服务监听
    server.serve_forever()
