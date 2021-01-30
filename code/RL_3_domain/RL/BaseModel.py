from abc import ABC, abstractmethod
import tensorflow as tf
import os
import numpy as np
import threading
lock = threading.Lock()

# 为了适应不同的强化学习任务，抽取的神经网络框架
class BaseModel(ABC):
    def __init__(self, StateShape, ACTIONS, extra_info):
        self._nActs = ACTIONS
        self._stateShape = StateShape
        self._debug = None
        self.__input_state, self.__pred_value = self._create_nn(extra_info)  # NN结构由子类实现
        self.__act_mask = tf.placeholder("float", [None, self._nActs])
        self.__real_value = tf.placeholder("float", [None])

        # define the cost function
        readout_action = tf.multiply(self.__pred_value, self.__act_mask)
        readout_action = tf.reduce_sum(readout_action, reduction_indices=1)
        self.__cost = tf.reduce_mean(tf.square(self.__real_value - readout_action))
        self.__train_op = tf.train.AdamOptimizer(1e-6).minimize(self.__cost)
        self.__saver = tf.train.Saver()
        # self.__sess = tf.InteractiveSession()
        self.__sess = tf.Session()
        self.__sess.run(tf.initialize_all_variables())


    # NN结构由子类实现，返回神经网络的输入和输出
    @abstractmethod
    def _create_nn(self, extra_info):
        pass


    def train(self, y_batch, a_batch, s_j_batch):
        # perform gradient step
        #print("training")

        #print(y_batch)

        fd = {
            self.__real_value: y_batch,
            self.__act_mask: a_batch,
            self.__input_state: s_j_batch}

        # 训练
        self.__train_op.run(session=self.__sess,feed_dict=fd)

        cost = self.__cost.eval(session=self.__sess,feed_dict=fd)
        # 计算loss
        return cost

    def loss(self, y_batch, a_batch, s_j_batch):
        fd = {
            self.__real_value: y_batch,
            self.__act_mask: a_batch,
            self.__input_state: s_j_batch}

        cost = self.__cost.eval(session=self.__sess, feed_dict=fd)
        # 计算loss
        return cost
        #print(cost)

        #w = self._debug[1]
        #self.__sess.run(tf.Print(w, [w], summarize=134))



    def pred(self, s_batch):

        global lock # 这段加锁的代码，我只是根据网上教程看的，不一定对，但应该没错
        #with self.__sess.as_default():
        lock.acquire()
        ret = self.__pred_value.eval(session=self.__sess, feed_dict={self.__input_state: s_batch})
        lock.release()

        return ret


    def save(self,path,t, extra_info, prt=False):
        pass
        if not os.path.exists(path):
            os.system('mkdir '+path)
        self.__saver.save(self.__sess, path+'/' + extra_info, global_step=t)
        if prt:
            print('NN参数成功保存')

    def load(self,path):
        # saving and loading networks


        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.__saver.restore(self.__sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        #exit()

    def get_nAct(self):
        return self._nActs


# 以下是工具函数的定义

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
