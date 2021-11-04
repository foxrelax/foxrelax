# -*- coding:utf-8 -*-
import os
import sys
import torch
from torch import nn

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.go.data.parallel_processor import GoDataProcessor
from foxrelax.go.encoder.oneplane import OnePlaneEncoder
from foxrelax.go.network import small
from foxrelax.ml import torch as ml


def run():
    go_board_rows, go_board_cols = 19, 19
    board_size = (go_board_rows, go_board_cols)
    num_classes = go_board_rows * go_board_cols
    num_train_games, num_test_games = 100, 100
    encoder = OnePlaneEncoder(board_size)
    processor = GoDataProcessor(encoder.name())
    train_generator = processor.load_go_data('train',
                                             num_train_games,
                                             use_generator=True)
    test_generator = processor.load_go_data('test',
                                            num_test_games,
                                            use_generator=True)
    print(
        f'训练样本的数量: {train_generator.get_num_samples()}, 测试样本的数量: {test_generator.get_num_samples()}'
    )

    # 构建网络模型
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = small.layers(input_shape, encoder.num_planes)
    network_layers.append(nn.Linear(512, num_classes))
    net = nn.Sequential(*network_layers)

    # 训练模型
    batch_size, num_epochs, lr = 256, 10, 0.1
    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    timer = ml.Timer()
    for epoch in range(num_epochs):
        # 训练损失之和, 训练准确率之和, 范例数
        metric = ml.Accumulator(3)
        net.train()
        # 每个epoch迭代一次, 所以每个epoch需要重新生成train_iter & test_iter
        train_iter, test_iter = train_generator._generate(
            batch_size), test_generator._generate(batch_size)
        for i, (X, y) in enumerate(train_iter):
            # X.shape - [256, 1, 19, 19]
            # y.shape - [256]
            timer.start()
            optimizer.zero_grad()
            y_hat = net(X)
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


if __name__ == "__main__":
    run()