# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from foxrelax.go.nn.load_mnist import load_data


def average_digit(data, digit):
    # x[0].shape - (784, 1)
    # x[1].shape - (10, 1)
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    # filtered_data.shape - (5851, 784, 1)
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8)

img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()