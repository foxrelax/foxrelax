# -*- coding:utf-8 -*-
from torch import nn


def layers(input_shape, in_channels):
    return [
        nn.ZeroPad2d(padding=2),
        nn.Conv2d(in_channels, 64, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.ZeroPad2d(padding=2),
        nn.Conv2d(64, 64, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.ZeroPad2d(padding=1),
        nn.Conv2d(64, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.ZeroPad2d(padding=1),
        nn.Conv2d(64, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.ZeroPad2d(padding=1),
        nn.Conv2d(64, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.Flatten(),

        # 中间的卷积层并没有改变input_shape, 所以全连接层的尺寸如下:
        nn.Linear(input_shape[-1] * input_shape[-2] * 64, 512),
        nn.ReLU()
    ]