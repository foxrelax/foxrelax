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
    batch_size, num_epochs, lr = 256, 15, 0.1
    net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=0),
        nn.Sigmoid(), nn.Conv2d(32,
                                64,
                                kernel_size=(3, 3),
                                stride=1,
                                padding=0), nn.Sigmoid(), nn.Flatten(),
        nn.Linear(1600, 256), nn.Sigmoid(), nn.Linear(256, 81), nn.Sigmoid())

    train_iter, test_iter = load_data_go(batch_size=batch_size)

    loss = nn.MSELoss()

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
            l = loss(y_hat, F.one_hot(y, 81).type(torch.float32))
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
    """
    可以看到, 这个模型的准确率并不高
    """
    # epoch 1, loss 0.087, train acc 0.012, test acc 0.014
    # epoch 2, loss 0.022, train acc 0.017, test acc 0.025
    # epoch 3, loss 0.016, train acc 0.025, test acc 0.025
    # epoch 4, loss 0.014, train acc 0.025, test acc 0.025
    # epoch 5, loss 0.014, train acc 0.025, test acc 0.025
    # epoch 6, loss 0.013, train acc 0.025, test acc 0.025
    # epoch 7, loss 0.013, train acc 0.025, test acc 0.025
    # epoch 8, loss 0.013, train acc 0.025, test acc 0.025
    # epoch 9, loss 0.013, train acc 0.025, test acc 0.025
    # epoch 10, loss 0.013, train acc 0.025, test acc 0.025
    # epoch 11, loss 0.013, train acc 0.025, test acc 0.025
    # epoch 12, loss 0.013, train acc 0.025, test acc 0.025
    # epoch 13, loss 0.012, train acc 0.025, test acc 0.025
    # epoch 14, loss 0.012, train acc 0.025, test acc 0.025
    # epoch 15, loss 0.012, train acc 0.025, test acc 0.025
    # 26283.0 examples/sec

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
    """
    这个simple CNN没有学到有用的知识
    """
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012
    # 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012 0.012