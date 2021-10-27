# -*- coding:utf-8 -*-
import numpy as np


def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)


def sigmoid_prime_double(x):
    """
    sigmoid梯度
    """
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)


# TODO
class Layer:
    def __init__(self):
        pass


# TODO
class ActivationLayer(Layer):
    def __init__(self):
        pass


# TODO
class DenseLayer(Layer):
    def __init__(self):
        pass