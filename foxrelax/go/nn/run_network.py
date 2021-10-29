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

    # 输出:
    # Initialize Network...
    # |--- DenseLayer
    #   |-- dimensions: (784,392)
    # |-- ActivationLayer
    #   |-- dimensions: (392,392)
    # |--- DenseLayer
    #   |-- dimensions: (392,196)
    # |-- ActivationLayer
    #   |-- dimensions: (196,196)
    # |--- DenseLayer
    #   |-- dimensions: (196,10)
    # |-- ActivationLayer
    #   |-- dimensions: (10,10)
    # Epoch 0: 3618 / 10000
    # Epoch 1: 5496 / 10000
    # Epoch 2: 5239 / 10000
    # Epoch 3: 6315 / 10000
    # Epoch 4: 6304 / 10000
    # Epoch 5: 6896 / 10000
    # Epoch 6: 7112 / 10000
    # Epoch 7: 7238 / 10000
    # Epoch 8: 7144 / 10000
    # Epoch 9: 7460 / 10000
