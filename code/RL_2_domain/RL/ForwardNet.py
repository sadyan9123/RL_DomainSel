from RL.BaseModel import BaseModel
from RL.BaseModel import weight_variable
from RL.BaseModel import bias_variable
from RL.BaseModel import conv2d
from RL.BaseModel import max_pool_2x2
import tensorflow as tf
from abc import ABC, abstractmethod

# 返回输出的张量，和该层的参数
def fc(input, output_dim):
    # 假设，输入的维度是 ？x INPUT_DIM
    input_dim = int(input.shape[1])
    W = weight_variable([input_dim, output_dim])
    b = bias_variable([output_dim])
    h = tf.matmul(input, W) + b
    return h, {"W":W, "b":b}

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

# 针对卷积类任务设计的神经网络
class ForwardNet3L(BaseModel):

    def _create_nn(self, extra_info):
        print('extra_info: ' + str(extra_info))
        # input layer

        # 输入是 [None, 4N]
        s = tf.placeholder("float", self._stateShape)
        input_dim = self._stateShape[1] * self._stateShape[2]
        s_reshape = tf.reshape(s, [-1, input_dim])
        hidden_size = 512

        h1, _ = fc(s_reshape, hidden_size)
        h2, _ = fc(h1, hidden_size)
        readout, _ = fc(h2, self._nActs)

        return s, readout

# 针对卷积类任务设计的神经网络
class ForwardNet5L(BaseModel):

    def _create_nn(self, extra_info):
        print('extra_info: ' + str(extra_info))
        # input layer

        # 输入是 [None, 4N]
        s = tf.placeholder("float", self._stateShape)
        input_dim = self._stateShape[1] * self._stateShape[2]
        s_reshape = tf.reshape(s, [-1, input_dim])
        hidden_size = 512

        h1, _ = fc(s_reshape, hidden_size)
        h2, _ = fc(h1, hidden_size)
        h3, _ = fc(h2, hidden_size*2)
        h4, _ = fc(h3, hidden_size)
        readout, _ = fc(h4, self._nActs)

        return s, readout

# 针对卷积类任务设计的神经网络
class ForwardNet5LReLU(BaseModel):

    def _create_nn(self, extra_info):
        print('extra_info: ' + str(extra_info))
        # input layer

        # 输入是 [None, 4N]
        s = tf.placeholder("float", self._stateShape)
        input_dim = self._stateShape[1] * self._stateShape[2]
        s_reshape = tf.reshape(s, [-1, input_dim])
        hidden_size = 512

        h1, _ = fc(s_reshape, hidden_size)
        h1 = relu(h1)
        h2, _ = fc(h1, hidden_size)
        h2 = relu(h2)
        h3, _ = fc(h2, hidden_size*2)
        h3 = relu(h3)
        h4, _ = fc(h3, hidden_size)
        readout, _ = fc(h4, self._nActs)

        return s, readout


# 针对卷积类任务设计的神经网络
class ForwardNet5LSigmoid(BaseModel):

    def _create_nn(self, extra_info):
        print('extra_info: '+str(extra_info))
        # input layer

        # 输入是 [None, 4N]
        s = tf.placeholder("float", self._stateShape)
        input_dim = self._stateShape[1] * self._stateShape[2]
        s_reshape = tf.reshape(s, [-1, input_dim])
        hidden_size = 512

        h1, _ = fc(s_reshape, hidden_size)
        h1 = sigmoid(h1)
        h2, _ = fc(h1, hidden_size)
        h2 = sigmoid(h2)
        h3, _ = fc(h2, hidden_size*2)
        h3 = sigmoid(h3)
        h4, _ = fc(h3, hidden_size)
        readout, _ = fc(h4, self._nActs)

        return s, readout



class ForwardNet3LConfigable(BaseModel):

    def _create_nn(self, extra_info):
        print('extra_info: ' + str(extra_info))

        h,a = extra_info.split('_')
        assert h[0]=='h' and a[0] == 'a'
        h = int(h[1:])
        a = a[1:]
        assert a in {'ReLU','Sigmoid','None'}, a

        print('ForwardNet3LConfigable::extra_info')

        print('\thidden size:'+str(h))
        print('\tactivation:'+a)

        # input layer

        # 输入是 [None, 4N]
        s = tf.placeholder("float", self._stateShape)
        input_dim = self._stateShape[1] * self._stateShape[2]
        s_reshape = tf.reshape(s, [-1, input_dim])
        #hidden_size = 512
        hidden_size = h

        h1, _ = fc(s_reshape, hidden_size)

        if a == 'ReLU':
            h1 = relu(h1)
            print('正在对h1加relu')
        if a == 'Sigmoid':
            h1 = sigmoid(h1)
            print('正在对h1加sigmoid')


        h2, _ = fc(h1, hidden_size)
        readout, _ = fc(h2, self._nActs)

        return s, readout