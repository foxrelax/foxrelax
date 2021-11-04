# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.ml import torch as ml
"""
在围棋中往往会有一些特定形状的局部棋子组合反复出现, 我们将它们称为`定式`. 为了能模仿人类的决策, 
围棋AI应该能够识别很多`局部定式`. 
"""


def load_data_go(batch_size=256):
    X = np.load(ml.download('go/features-40k'))  # X.shape - (41439, 1, 9, 9)
    Y = np.load(ml.download('go/labels-40k'))  # Y.shape - (41439, 81)
    samples = X.shape[0]
    size = 9
    X = X.reshape(samples, 1, size, size)  # X.shape - (41439, 1, 9, 9)
    Y = Y.reshape(samples, size * size).argmax(axis=1)  # Y.shape - (41439,)
    train_samples = int(0.9 * samples)
    X_train, X_test = X[:train_samples], X[train_samples:]
    Y_train, Y_test = Y[:train_samples], Y[train_samples:]
    return ml.load_array(
        (torch.tensor(X_train).type(torch.float32), torch.tensor(Y_train)),
        batch_size,
        is_train=True), ml.load_array(
            (torch.tensor(X_test).type(torch.float32), torch.tensor(Y_test)),
            batch_size,
            is_train=True)


def run():
    """
    直接运行
    """
    batch_size, num_epochs, lr = 256, 25, 0.5
    net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)), nn.Flatten(), nn.Linear(256, 128),
        nn.ReLU(), nn.Linear(128, 81))

    train_iter, test_iter = load_data_go(batch_size=batch_size)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    timer = ml.Timer()
    for epoch in range(num_epochs):
        # 训练损失之和, 训练准确率之和, 范例数
        metric = ml.Accumulator(3)
        net.train()
        for _, (X, y) in enumerate(train_iter):
            # X.shape - [256, 1, 9, 9]
            # y.shape - [256]
            timer.start()
            optimizer.zero_grad()
            y_hat = net(X)
            # 由于使用MSE Loss, 这里需要做一次one hot encoding
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], ml.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = ml.evaluate_accuracy(net, test_iter)
        print(
            f'epoch {epoch+1}, loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')
    return net


if __name__ == '__main__':
    net = run()
    # epoch 1, loss 4.244, train acc 0.024, test acc 0.027
    # epoch 2, loss 4.090, train acc 0.028, test acc 0.009
    # epoch 3, loss 4.020, train acc 0.033, test acc 0.015
    # epoch 4, loss 3.982, train acc 0.037, test acc 0.018
    # epoch 5, loss 3.956, train acc 0.041, test acc 0.017
    # epoch 6, loss 3.924, train acc 0.045, test acc 0.014
    # epoch 7, loss 3.887, train acc 0.053, test acc 0.016
    # epoch 8, loss 3.848, train acc 0.057, test acc 0.016
    # epoch 9, loss 3.809, train acc 0.062, test acc 0.012
    # epoch 10, loss 3.775, train acc 0.068, test acc 0.014
    # epoch 11, loss 3.744, train acc 0.071, test acc 0.013
    # epoch 12, loss 3.712, train acc 0.078, test acc 0.014
    # epoch 13, loss 3.678, train acc 0.082, test acc 0.018
    # epoch 14, loss 3.652, train acc 0.088, test acc 0.014
    # epoch 15, loss 3.624, train acc 0.094, test acc 0.014
    # epoch 16, loss 3.594, train acc 0.099, test acc 0.018
    # epoch 17, loss 3.569, train acc 0.103, test acc 0.021
    # epoch 18, loss 3.539, train acc 0.108, test acc 0.013
    # epoch 19, loss 3.513, train acc 0.114, test acc 0.016
    # epoch 20, loss 3.482, train acc 0.120, test acc 0.022
    # epoch 21, loss 3.455, train acc 0.124, test acc 0.014
    # epoch 22, loss 3.433, train acc 0.128, test acc 0.022
    # epoch 23, loss 3.402, train acc 0.133, test acc 0.025
    # epoch 24, loss 3.381, train acc 0.140, test acc 0.017
    # epoch 25, loss 3.357, train acc 0.145, test acc 0.022

    # 预测
    test_board = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, -1, 1, -1, 0, 0, 0, 0],
                               [0, 1, -1, 1, -1, 0, 0, 0, 0],
                               [0, 0, 1, -1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0,
                                0]]).type(torch.float32)
    move_probs = net(test_board.reshape(1, 1, 9, 9))[0]
    move_probs = F.softmax(move_probs)
    i = 0
    for row in range(9):
        row_formatted = []
        for col in range(9):
            row_formatted.append('{:.3f}'.format(move_probs[i]))
            i += 1
        print(' '.join(row_formatted))
    # 0.000 0.000 0.000 0.001 0.001 0.000 0.000 0.000 0.000
    # 0.000 0.005 0.010 0.025 0.028 0.011 0.007 0.001 0.000
    # 0.000 0.006 0.000 0.103 0.096 0.038 0.000 0.002 0.000
    # 0.001 0.002 0.005 0.041 0.060 0.030 0.008 0.002 0.000
    # 0.002 0.004 0.006 0.029 0.007 0.037 0.012 0.004 0.002
    # 0.000 0.004 0.013 0.096 0.069 0.061 0.013 0.003 0.000
    # 0.001 0.004 0.000 0.046 0.021 0.013 0.000 0.002 0.000
    # 0.001 0.003 0.010 0.005 0.027 0.005 0.008 0.001 0.000
    # 0.000 0.000 0.000 0.001 0.000 0.000 0.000 0.000 0.000