# -*- coding:utf-8 -*-
import random
import numpy as np


class MSE:
    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels):
        """
        MSE Error
        """
        diff = predictions - labels
        return 0.5 * sum(diff * diff)[0]

    @staticmethod
    def loss_derivative(predictions, labels):
        """
        MSE Loss的导数
        """
        return predictions - labels


class SequentialNetwork:
    """
    顺序神经网络可以按顺序堆叠多个层
    """
    def __init__(self, loss=None):
        print('Initialize Network...')
        self.layers = []
        if loss is None:
            self.loss = MSE()
        else:
            self.loss = loss

    def add(self, layer):
        """
        每添加一层, 都需要将它与前导层连接起来, 并输出层的描述
        """
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(self,
              training_data,
              epochs,
              mini_batch_size,
              learning_rate,
              test_data=None):
        """
        当前这个版本训练起来是非常慢的, 没有用到任何GPU硬件加速的特性, 针对小批量的训练的时候,
        也是迭代小批量中每个样本单独计算, 效率很低
        """
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                # 对每个小批量进行训练
                self.train_batch(mini_batch, learning_rate)
            if test_data:
                # 如果提供了测试数据, 就在每个迭代周期结束后评估网络
                n_test = len(test_data)
                print('Epoch {0}: {1} / {2}'.format(epoch,
                                                    self.evaluate(test_data),
                                                    n_test))
            else:
                print('Epoch {0} complete'.format(epoch))

    def train_batch(self, mini_batch, learning_rate):
        """
        训练一个小批量

        1. 前向传播 & 后向传播
        2. 更新参数
        """
        self.forward_backward(mini_batch)
        self.update(mini_batch, learning_rate)

    def update(self, mini_batch, learning_rate):
        """
        更新参数
        """
        # 用小批量尺寸来归一化学习率是一个技巧, 可以控制更新规模
        # 因为小批量尺寸越大, 累计的delta就越大, 可以用这个技巧来控制更新
        learning_rate = learning_rate / len(mini_batch)
        for layer in self.layers:
            layer.update_params(learning_rate)
        for layer in self.layers:
            layer.clear_deltas()

    def forward_backward(self, mini_batch):
        """
        前向传播 & 后向传播
        """
        for x, y in mini_batch:
            # 前向传播
            # x.shape - (784, 1)
            # y.shape - (10, 1)
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()

            # 后向传播
            self.layers[-1].input_delta = self.loss.loss_derivative(
                self.layers[-1].output_data, y)
            for layer in reversed(self.layers):
                layer.backward()

    def single_forward(self, x):
        """
        前向传播单个样本的数据, 并返回结果
        """
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):
        """
        计算测试数据上的预测准确率
        """
        test_results = [(np.argmax(self.single_forward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)