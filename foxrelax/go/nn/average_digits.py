# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.go.nn.load_mnist import load_data
from foxrelax.go.nn.layers import sigmoid_double


def average_digit(data, digit):
    """
    计算digit的平均图像

    对手写数字进行粗略的分类: 给定一个数字8的图像, 那么与其它数字相比, 它应该更接近于8的平均图像
    """
    # x[0].shape - (784, 1)
    # x[1].shape - (10, 1)
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    # filtered_data.shape - (5851, 784, 1)
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


def predict(x, W, b):
    return sigmoid_double(np.dot(W, x) + b)


def evaluate(data, digit, threshold, W, b):
    """
    评估使用了决策阀值的模型预测`准确率`
    """
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:
            # 将数字8的实例预测为8是一次正确的预测
            correct_predictions += 1
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:
            # 如果预测值低于设定的阀值, 并且样本也确实不是数字8, 那么这也是一次正确预测
            correct_predictions += 1
    return correct_predictions / total_samples


if __name__ == '__main__':
    # 显示数字8的平均图像, 平均图像会有点`模糊`
    train, test = load_data()
    avg_eight = average_digit(train, 8)
    img = (np.reshape(avg_eight, (28, 28)))
    plt.imshow(img)
    plt.show()

    # avg_eight包含了数字8在图像中如何呈现的大量相关信息, 可以使用avg_eight作为一个简单模型的参数(权重),
    # 判断某个数字的输入向量x是否为8.
    # W = evg_eight.T, 然后计算W和x的点积, 会将W和x的像素逐对相乘, 并将784个结果值相加. 如果x确实是数字8,
    # 那么x的像素在与W相同的地方应该有差不多的色调值. 相反, 如果x不是数字8, 那么它与W的重叠就会比较少
    #
    # 我们选取mnist中的两个样本, 分别是数字4和8, 结果是数字8的点积是54.1, 数字4的点积是20.0. 符合预期
    x_3 = train[2][0]  # 数字4
    x_18 = train[17][0]  # 数字8
    W = np.transpose(avg_eight)  # W.shape - (1, 784)
    print(np.dot(W, x_3))  # [[19.93822415]]
    print(np.dot(W, x_18))  # [[54.10696465]]

    # 我们通过sigmoid函数将输出范围压缩到[0,1]. 将输出结果转化为概率值.
    # 由于sigmod(54.1)和sigmoid(20.0)都接近1了. 这时候需要一个偏差项(bias), 这里bias设置为-45
    b = -45
    print(predict(x_3, W, b))  # [[1.30559669e-11]]
    print(predict(x_18, W, b))  # [[0.99988912]]

    # 模型现在只能区分特定的数字(此处为数字8)与其它数字, 由于训练集和测试集中每个数字的图像数量分布是均衡的,
    # 数字8的样本大约只占10%, 因此, 只需要一个始终预测为0的简单模型, 就能得到大约90%的准确率. 在分析解决分类
    # 问题的时候, 需要特别注意这种`分类不均匀`的情况

    # 在训练集上的准确率
    print(evaluate(data=train, digit=8, threshold=0.5, W=W, b=b))  # 0.690
    # 在测试集上的准确率
    print(evaluate(data=test, digit=8, threshold=0.5, W=W, b=b))  # 0.678
    # 只针对测试集数字8的准确率
    eight_test = [x for x in test if np.argmax(x[1]) == 8]
    print(evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b))  # 0.810
