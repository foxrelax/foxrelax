# -*- coding:utf-8 -*-
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from foxrelax.go.nn import load_mnist
from foxrelax.go.nn import network
from foxrelax.go.nn.layers import (DenseLayer, ActivationLayer)

if __name__ == '__main__':
    # 加载训练数据和测试数据
    training_data, test_data = load_mnist.load_data()

    # 初始化顺序神经网络
    net = network.SequentialNetwork()

    net.add(DenseLayer(784, 392))
    net.add(ActivationLayer(392))
    net.add(DenseLayer(392, 196))
    net.add(ActivationLayer(196))
    net.add(DenseLayer(196, 10))
    net.add(ActivationLayer(10))

    # 训练
    net.train(training_data,
              epochs=10,
              mini_batch_size=10,
              learning_rate=3.0,
              test_data=test_data)
