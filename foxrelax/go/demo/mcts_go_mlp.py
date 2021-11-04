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


def load_data_go(batch_size=256):
    X = np.load(ml.download('go/features-40k'))  # X.shape - (41439, 1, 9, 9)
    Y = np.load(ml.download('go/labels-40k'))  # Y.shape - (41439, 81)
    samples = X.shape[0]
    board_size = 9 * 9
    X = X.reshape(samples, board_size)  # X.shape - (41439, 81)
    Y = Y.reshape(samples, board_size).argmax(axis=1)  # Y.shape - (41439,)
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
    net = nn.Sequential(nn.Flatten(), nn.Linear(81, 512), nn.Sigmoid(),
                        nn.Linear(512, 256), nn.Sigmoid(), nn.Linear(256, 81),
                        nn.Sigmoid())

    train_iter, test_iter = load_data_go(batch_size=batch_size)

    loss = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    timer = ml.Timer()
    for epoch in range(num_epochs):
        # 训练损失之和, 训练准确率之和, 范例数
        metric = ml.Accumulator(3)
        net.train()
        for _, (X, y) in enumerate(train_iter):
            # X.shape - [256, 81]
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
    """
    对于9x9的棋盘, 一共81种可能的落子动作, 因此网络需要81个分类, 如果我们完全随机的预测,
    机会有1/81, 也就是1.2%的准确率. 我们这个网络经过训练后, 可以达到2.3%的准确率, 这并
    不令人满意, 但是它确实学习到了一些东西, 预测效果比随机要好一些
    """
    net = run()
    # epoch 1, loss 0.096, train acc 0.016, test acc 0.016
    # epoch 2, loss 0.026, train acc 0.017, test acc 0.024
    # epoch 3, loss 0.018, train acc 0.025, test acc 0.025
    # epoch 4, loss 0.016, train acc 0.026, test acc 0.025
    # epoch 5, loss 0.015, train acc 0.026, test acc 0.025
    # epoch 6, loss 0.014, train acc 0.026, test acc 0.025
    # epoch 7, loss 0.014, train acc 0.026, test acc 0.025
    # epoch 8, loss 0.013, train acc 0.026, test acc 0.025
    # epoch 9, loss 0.013, train acc 0.026, test acc 0.025
    # epoch 10, loss 0.013, train acc 0.026, test acc 0.025
    # epoch 11, loss 0.013, train acc 0.026, test acc 0.025
    # epoch 12, loss 0.013, train acc 0.026, test acc 0.025
    # epoch 13, loss 0.013, train acc 0.026, test acc 0.025
    # epoch 14, loss 0.013, train acc 0.026, test acc 0.025
    # epoch 15, loss 0.013, train acc 0.026, test acc 0.025
    # 198917.1 examples/sec

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
    move_probs = net(test_board.reshape(1, 9, 9))[0]
    i = 0
    for row in range(9):
        row_formatted = []
        for col in range(9):
            row_formatted.append('{:.3f}'.format(move_probs[i]))
            i += 1
        print(' '.join(row_formatted))
    """
    输出的9x9矩阵, 每个数字代表模型在棋盘上这个点上下一回合落子的置信度. 模型输出
    的结果并不太好. 它甚至连不能在被棋子占据的地方落子都没有学会. 但是棋盘的边缘得分
    始终低于靠近中心的得分. 而根据围棋的传统, 除终盘或者其它特殊情况, 应该尽量避免在
    棋盘边缘落子. 这样看来, 我们的模型已经学会了一个围棋相关的合理概念, 这已经算进步了

    我们还面临以下一些问题:
    1. 这个预测模型使用的数据是由树搜索算法生成的, 而这个算法的随机性很高, 有时候MCTS
    引擎会产生很奇怪的动作, 尤其在遥遥领先或者远远落后的局面下. 人类的策略也有出乎意料的
    时候, 但他们至少不会下一些毫无道理的废棋
    2. 这个例子把9x9的棋盘拉平了, 都是了全部的空间信息
    """
    # 0.028 0.029 0.029 0.030 0.030 0.030 0.029 0.029 0.028
    # 0.029 0.032 0.033 0.034 0.035 0.034 0.032 0.031 0.029
    # 0.029 0.032 0.029 0.036 0.034 0.036 0.029 0.032 0.029
    # 0.030 0.034 0.036 0.037 0.034 0.034 0.032 0.034 0.030
    # 0.030 0.034 0.037 0.034 0.034 0.034 0.037 0.034 0.030
    # 0.029 0.033 0.035 0.037 0.038 0.034 0.036 0.033 0.029
    # 0.029 0.032 0.029 0.036 0.037 0.036 0.029 0.032 0.030
    # 0.029 0.031 0.032 0.033 0.034 0.034 0.032 0.032 0.029
    # 0.027 0.029 0.029 0.030 0.030 0.029 0.030 0.028 0.027
