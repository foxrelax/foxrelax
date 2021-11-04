# -*- coding:utf-8 -*-
import os
import sys
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.ml import torch as ml


def run():
    """
    直接运行
    """
    batch_size, num_epochs, lr = 256, 10, 0.1
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 10))

    train_iter, test_iter = ml.load_data_mnist(batch_size=batch_size)
    loss = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    timer = ml.Timer()
    for epoch in range(num_epochs):
        # 训练损失之和, 训练准确率之和, 范例数
        metric = ml.Accumulator(3)
        net.train()
        for _, (X, y) in enumerate(train_iter):
            # X.shape - [256, 1, 28, 28]
            # y.shape - [256]
            timer.start()
            optimizer.zero_grad()
            y_hat = net(X)
            # 由于使用MSE Loss, 这里需要做一次one hot encoding
            l = loss(y_hat, F.one_hot(y, 10).type(torch.float32))
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
    run()
    # epoch 1, loss 0.060, train acc 0.703, test acc 0.835
    # epoch 2, loss 0.042, train acc 0.845, test acc 0.870
    # epoch 3, loss 0.036, train acc 0.869, test acc 0.890
    # epoch 4, loss 0.032, train acc 0.884, test acc 0.901
    # epoch 5, loss 0.029, train acc 0.896, test acc 0.908
    # epoch 6, loss 0.027, train acc 0.903, test acc 0.914
    # epoch 7, loss 0.025, train acc 0.910, test acc 0.920
    # epoch 8, loss 0.024, train acc 0.915, test acc 0.925
    # epoch 9, loss 0.022, train acc 0.920, test acc 0.928
    # epoch 10, loss 0.022, train acc 0.923, test acc 0.931
    # 267289.9 examples/sec