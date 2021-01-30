import numpy as np
import random
from collections import deque
import os
import sys


class QLeaner:
    def __init__(self, NET, logPath, gamma, epoch, batch_size):
        print("QLeaner 正在构造")
        self.__net = NET
        self.__nAct = NET.get_nAct()
        self.__FRAME_PER_ACTION = 1
        self.__t = 0
        # self.__OBSERVE = 100000. # timesteps to observe before training
        self.__OBSERVE = 100.  # timesteps to observe before training
        self.__ε_final = 0.1  # final value of epsilon
        # self.__ε_init = 0.0001  # starting value of epsilon
        self.__ε_init = 1  # starting value of epsilon
        self.__ε = self.__ε_init

        self.__EXPLORE = 20000.  # frames over which to anneal epsilon 退火所需要的帧数
        self.__REPLAY_MEMORY = 100000  # number of previous transitions to remember
        #self.__BATCH_SIZE = 128 * 16  # size of minibatch
        self.__BATCH_SIZE = int(batch_size) if batch_size is not None else None  # size of minibatch

        # 测试模式下，不需要配置epoch和gamma
        self.__EPOCH = int(epoch) if epoch is not None else None
        #self.__γ = 0.9  # decay rate of past observations
        #self.__γ = 0.0  # decay rate of past observations # 只看当前reward
        self.__γ = float(gamma) if gamma is not None else None

        # store the previous observations in replay memory
        # store the transition in D
        self.__TransMem = deque()
        self.__logPath = logPath
        self.__net.load(logPath)
        self.__global_training_loss_history = list()

        print("QLeaner配置")
        print("\t备忘录大小: " + str(self.__REPLAY_MEMORY))
        print("\tBatch大小:" + str(self.__BATCH_SIZE))
        print("\tγ:"+str(self.__γ))
        print("\tepoch:"+str(self.__EPOCH))

    def pred_action(self, state, noRand = False):

        # choose an action epsilon greedily

        pred_t = self.__net.pred([state])[0]
        # print('state: ',state)
        # print('readout: ', pred_t)
        a_t = np.zeros([self.__nAct])
        action_index = 0

        # 如果没有随机性，就直接返回NN的结果
        if noRand:
            action_index = np.argmax(pred_t)
            a_t[action_index] = 1
            return a_t

        if self.__t % self.__FRAME_PER_ACTION == 0:
            if random.random() <= self.__ε:
                print("----------Random Action----------")
                action_index = random.randrange(self.__nAct)
            else:
                action_index = np.argmax(pred_t)
        a_t[action_index] = 1

        # scale down epsilon
        if self.__ε > self.__ε_final and self.__t > self.__OBSERVE:
            self.__ε -= (self.__ε_init - self.__ε_final) / self.__EXPLORE

        return a_t

    def add_trans(self, s_t, a_t, r_t, s_t1, terminal):

        # print("ε: ",self.__ε)

        # store the transition in D
        trans = (s_t, a_t, r_t, s_t1, terminal)
        self.__TransMem.append(trans)

        if len(self.__TransMem) > self.__REPLAY_MEMORY:
            self.__TransMem.popleft()
            #print('备忘录空间不足，删除旧trans。备忘录空间为 ' + str(self.__REPLAY_MEMORY))
            print('备忘录空间不足')
            assert False, '备忘录空间不足'

    # 随机梯度下降
    # 注意：时间戳以训练的时间为准
    def sgd(self):

        # 保存神经网络数据，放在第一行是为了显示初始loss
        if self.__t % 100 == 0:

            #print("NN已成功保存第"+str(self.__t)+'轮数据')
            curr_gl=self.global_training_loss()

            if len(self.__global_training_loss_history)>0:
                last_min_gl = min(self.__global_training_loss_history)
                last_min_gl_idx = 100*np.argmin(self.__global_training_loss_history)
            else:
                last_min_gl = float('inf')
                last_min_gl_idx = None


            self.__global_training_loss_history.append(curr_gl)



            # TODO 这里是否应该检测 best global train loss 再去保存？
            # TODO 先这么实现了吧。。。。
            if curr_gl < last_min_gl:
                self.__net.save(self.__logPath, self.__t, '{:.2f}'.format(curr_gl))
                para_msg = 'save'
            else:
                para_msg = 'drop'

            print('\rcurr global train loss: {:.5f} @ {}  # last min {:.5f} @ {} {}'.format(curr_gl, self.__t, last_min_gl,
                                                                                            last_min_gl_idx, para_msg))
            sys.stdout.flush()

            if self.__t >= self.__EPOCH:
                print("已完成全部"+str(self.__EPOCH)+"轮训练，训练结束")
                sys.stdout.flush()
                exit()

        # sample __act_mask minibatch to train on
        minibatch = random.sample(self.__TransMem, self.__BATCH_SIZE)

        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]

        y_batch = []

        readout_j1_batch = self.__net.pred(s_j1_batch)

        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            # if terminal, only equals reward
            if terminal:
                y_batch.append(r_batch[i])
            else:
                # 由reaward计算出value
                y_batch.append(r_batch[i] + self.__γ * np.max(readout_j1_batch[i]))

        # # perform gradient step

        # loss_old = self.__net.loss(y_batch, a_batch, s_j_batch)

        loss = self.__net.train(y_batch, a_batch, s_j_batch)

        #print('\rloss: {:.3f} -> {:.3f}'.format(loss_old, loss))
        self.__t += 1




    # TODO 算全局loss
    def global_training_loss(self):

        # sample __act_mask minibatch to train on
        # minibatch = random.sample(self.__TransMem, self.__BATCH_SIZE)
        minibatch = self.__TransMem

        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]

        y_batch = []

        readout_j1_batch = self.__net.pred(s_j1_batch)

        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            # if terminal, only equals reward
            if terminal:
                y_batch.append(r_batch[i])
            else:
                # 由reaward计算出value
                y_batch.append(r_batch[i] + self.__γ * np.max(readout_j1_batch[i]))

        # # perform gradient step

        gl = self.__net.loss(y_batch, a_batch, s_j_batch)
        return gl

