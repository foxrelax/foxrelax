# -*- coding:utf-8 -*-
import os
import sys
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.ml import torch as ml

np.random.seed(123)
X = np.load(ml.download('go/features-200'))
Y = np.load(ml.download('go/labels-200'))
print(X.shape, Y.shape)