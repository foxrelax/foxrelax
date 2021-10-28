# -*- coding:utf-8 -*-
import numpy as np


def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)


def sigmoid_prime_double(x):
    """
    sigmoid梯度
    """
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)


class Layer:
    """
    将多个Layer堆叠起来构建顺序神经网络

    层不仅要包含输入数据的算法(前向传播), 还需要包含将误差进行反向传播的机制.
    为了在反向传播过程中不用重新计算激活值, 我们需要维护双向传递进出该层的数据状态.
    """
    def __init__(self):
        self.params = []
        self.previous = None  # Layer的前一层
        self.next = None  # Layer的后一层

        # 每一层都可以保留前向传播中流入和流出该层的数据
        self.input_data = None
        self.output_data = None

        # 每一层都可以保留后向传播流入和流出该层的数据
        self.input_delta = None
        self.output_delta = None

    def connect(self, layer):
        """
        将当前层加入到神经网络当前最后的位置
        """
        self.previous = layer
        layer.next = self

    def forward(self):
        """
        前向传播
        """
        raise NotImplementedError

    def get_forward_input(self):
        """
        第一层不做处理; 其他层的input_data是从前一层
        的输出中获取
        """
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):
        """
        后向传播
        """
        raise NotImplementedError

    def get_backward_input(self):
        """
        最后一层不做处理; 其他层都会从他的后继层获取误差项
        """
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    def clear_deltas(Self):
        """
        为每一个小批量的数据计算各种增量并累积起来. 在计算下一个小批量开始时,
        要重置所有的增量
        """
        pass

    def update_params(self, learning_rate):
        """
        使用指定的学习率, 根据当前的各种增量来更新当前层的参数
        """
        pass

    def describe(self):
        """
        描述自身的方法
        """
        raise NotImplementedError


class ActivationLayer(Layer):
    """
    sigmoid激活层
    """
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        """
        前向传播只需要将sigmoid激活函数应用于输入数据即可
        """
        data = self.get_forward_input()
        self.output_data = sigmoid(data)

    def backward(self):
        """
        反向传播需要将`输入数据的sigmoid函数的导数`与`误差项`逐个元素相乘
        """
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)

    def describe(self):
        print('|-- ' + self.__class__.__name__)
        print('  |-- dimensions: ({},{})'.format(self.input_dim,
                                                 self.output_dim))


class DenseLayer(Layer):
    """
    全连接层需要指定输入和输出的维度
    """
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 随机初始化w, b
        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]

        # w, b的增量设置为0
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        delta = self.get_backward_input()  # delta.shape - (output_dim, 1)
        data = self.get_forward_input()  # delta.shape - (input_dim, 1)

        # 更新w, b的增量
        self.delta_b += delta  # delta_b.shape - (output_dim, 1)
        self.delta_w += np.dot(
            delta, data.transpose())  # delta_w.shape - (output_dim, input_dim)

        # 将输出增量传递到前一层, 完成反向传播
        self.output_delta = np.dot(
            self.weight.transpose(),
            delta)  # output_delta.shape - (intput_dim, 1)

    def update_params(self, rate):
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):
        print('|--- ' + self.__class__.__name__)
        print('  |-- dimensions: ({},{})'.format(self.input_dim,
                                                 self.output_dim))
