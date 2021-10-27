# -*- coding:utf-8 -*-
import numpy as np
import foxrelax.ml.torch as ml


def encode_label(j):
    """
    one-hot encoding

    将索引j编码为长度为10的向量
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def shape_data(data):
    # 将输入的28x28的图像拉伸成784的特征向量
    features = [np.reshape(x, (784, 1)) / 256 for x in data[0]]
    # 所有的标签使用one-hot encoding
    labels = [encode_label(y) for y in data[1]]
    return list(zip(features, labels))


def load_data_impl():
    path = ml.download('mnist')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def load_data():
    """
    train, test = load_data()
    for X, y in train:
        print(X.shape, y.shape)  # (784, 1) (10, 1)
        break
    """
    train_data, test_data = load_data_impl()
    return shape_data(train_data), shape_data(test_data)