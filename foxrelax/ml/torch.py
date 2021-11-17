# -*- coding:utf-8 -*-
import time
import os
import re
import collections
import random
import math
import hashlib
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

DATA_URL = 'http://oss.foxrelax.com'
DATA_HUB = dict()
DATA_HUB['logistic_regression_unseparable'] = (
    'csv', '/ml/logistic_regression_unseparable.csv',
    'f7a9c08c4b11fa8e03367a996ac75e5539e25c36',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/logistic_regression_unseparable.csv'
)
DATA_HUB['logistic_regression_separable'] = (
    'csv', '/ml/logistic_regression_separable.csv',
    '97ff298c6aed70329f6f28757de04759f5600956',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/logistic_regression_separable.csv'
)
DATA_HUB['airfoil_self_noise'] = (
    'dat', '/ml/airfoil_self_noise.dat',
    '76e5be1548fd8222e5074cf0faae75edff8cf93f',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/airfoil_self_noise.dat')
DATA_HUB['mnist'] = (
    'npz', '/ml/mnist.npz', 'e7727b719822ece6d78ca83af078bcd85ae81b0d',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/mnist.npz')
DATA_HUB['kaggle_house_train'] = (
    'csv', '/ml/kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/kaggle_house_pred_train.csv'
)
DATA_HUB['kaggle_house_test'] = (
    'csv', '/ml/kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/kaggle_house_pred_test.csv'
)
DATA_HUB['time_machine'] = (
    'txt', '/ml/time_machine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/time_machine.txt')
DATA_HUB['fra_eng'] = (
    'zip', '/ml/fra_eng.zip', '02eee9efbc64e076be914c6c163740dd5d448b36',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/fra_eng.zip')
DATA_HUB['hotdog'] = (
    'zip', '/ml/hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/hotdog.zip')
DATA_HUB['banana_detection'] = (
    'zip', '/ml/banana_detection.zip',
    '068e13f04d30c8b96d30e490394ce9ab7cbdf2d5',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/banana_detection.zip')
DATA_HUB['img/cat1'] = (
    'jpg', '/ml/img/cat1.jpg', 'f9c5b905d908b97eeeb64ff34a46fa8b143f88f8',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/cat1.jpg')
DATA_HUB['img/cat2'] = (
    'jpg', '/ml/img/cat2.jpg', 'b712adcb9ca6af53081bd96426b719573f40053e',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/cat2.jpg')
DATA_HUB['img/cat3'] = (
    'jpg', '/ml/img/cat3.jpg', '80249a6aa841706d861f3f7efad582f6828ca3d0',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/cat3.jpg')
DATA_HUB['img/catdog'] = (
    'jpg', '/ml/img/catdog.jpg', '60b7d540db03eef6b9834329bccc4417ef349bf6',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/catdog.jpg')
DATA_HUB['go/features-20'] = (
    'npy', '/ml/go/features-20.npy',
    '3f84459adb1d5dd66a6395dfbc513b99349fd8f0',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/features-20.npy')
DATA_HUB['go/features-200'] = (
    'npy', '/ml/go/features-200.npy',
    'cb8628ffc84c9b6ecb1bd0991255b7bb3d85b0d7',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/features-200.npy')
DATA_HUB['go/features-40k'] = (
    'npy', '/ml/go/features-40k.npy',
    'f3f0bdb3dd8a5cc663ef56ca4c8e06032034531d',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/features-40k.npy')
DATA_HUB['go/labels-20'] = (
    'npy', '/ml/go/labels-20.npy', 'c6f03c99916218175e185601530fc6a32d26c114',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/labels-20.npy')
DATA_HUB['go/labels-200'] = (
    'npy', '/ml/go/labels-200.npy', 'ef7789e4b68555c903bc66ec56ecbf7af956d081',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/labels-200.npy')
DATA_HUB['go/labels-40k'] = (
    'npy', '/ml/go/labels-40k.npy', 'd959a562ef5189413a0d1b3525972e9c9dd2b598',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/labels-40k.npy')

DATA_HUB['dlwpt/audio-chirp'] = (
    'zip', '/ml/dlwpt/audio-chirp.zip',
    '9f76a2703b2efc92d942a92e46acea76e6059db8',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/audio-chirp.zip')
DATA_HUB['dlwpt/bike-sharing-dataset'] = (
    'zip', '/ml/dlwpt/bike-sharing-dataset.zip',
    'fc8e284c60658bb4121782a424cbc4e955ebd3d6',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/bike-sharing-dataset.zip'
)
DATA_HUB['dlwpt/image-cats'] = (
    'zip', '/ml/dlwpt/image-cats.zip',
    '45f71d8fca66d81f9a0218ed0df87caf8beb5962',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/image-cats.zip')
DATA_HUB['dlwpt/image-dog'] = (
    'zip', '/ml/dlwpt/image-dog.zip',
    '13d7bfdf4a1ad781977fb3ffe85a21a8dc18ea84',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/image-dog.zip')

DATA_HUB['dlwpt/jane-austen'] = (
    'zip', '/ml/dlwpt/jane-austen.zip',
    'da49eb304b63fff1f3a385e75f8a2076a9b823a6',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/jane-austen.zip')
DATA_HUB['dlwpt/tabular-wine'] = (
    'zip', '/ml/dlwpt/tabular-wine.zip',
    'e5c949b1bf4c47bd41a85c950d96a3a16a0d608b',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/tabular-wine.zip')
DATA_HUB['dlwpt/video-cockatoo'] = (
    'zip', '/ml/dlwpt/video-cockatoo.zip',
    '71213219987ecd54e40ce5154cf28f7ab8d4ea91',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/video-cockatoo.zip'
)
DATA_HUB['dlwpt/volumetric-dicom'] = (
    'zip', '/ml/dlwpt/volumetric-dicom.zip',
    'd6689fd995a22328db45d969a22a364c27dc9a29',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/dlwpt/volumetric-dicom.zip'
)


def download(name, cache_dir=os.path.join('..', 'data')):
    assert name in DATA_HUB, f"{name} 不存在"
    _, _, sha1_hash, url = DATA_HUB[name]
    fname = os.path.join(cache_dir, url.split('/ml/')[-1])
    fdir = os.path.dirname(fname)
    os.makedirs(fdir, exist_ok=True)
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f'下载{fname}完成!')
    return fname


def extract(filename, folder=None):
    """Download and extract a zip/tar file"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)


def download_extract(name, folder=None):
    """Download and extract a zip/tar file"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def load_data(name, cache_dir=os.path.join('..', 'data')):
    """load ml data."""
    return pd.read_csv(download(name, cache_dir))


def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(6, 4)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X,
         Y=None,
         xlabel=None,
         ylabel=None,
         legend=None,
         xlim=None,
         ylim=None,
         xscale='linear',
         yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(6, 4),
         axes=None):
    """
    画图(可以同时画多条曲线)
    
    Usage:
    # 生成数据
    x = np.linspace(1, 10, 100)
    y1, y2 = np.sin(x), np.cos(x)

    # 画图
    ml.plot(y1)  # 使用0-100作为作为x轴来画图
    ml.plot([y1, y2])  # 使用0-100作为作为x轴来画图
    ml.plot(x, y1)
    ml.plot(x, [y1, y2])
    ml.plot([x, x], [y1, y2])
    """
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, 'ndim') and X.ndim == 1
                or isinstance(X, list) and not hasattr(X[0], '__len__'))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和, 样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def sgd(params, lr, batch_size):
    """小批量梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers(num_workers=2):
    """Use multi processes to read the data"""
    return num_workers


def load_data_mnist(batch_size, resize=None, root='../data'):
    """下载MNIST数据集, 然后将其加载到内存中"""
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在0到1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root=root,
                                             train=True,
                                             transform=trans,
                                             download=True)
    mnist_test = torchvision.datasets.MNIST(root=root,
                                            train=False,
                                            transform=trans,
                                            download=True)
    return (data.DataLoader(mnist_train,
                            batch_size,
                            shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,
                            batch_size,
                            shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_fashion_mnist(batch_size, resize=None, root='../data'):
    """下载Fashion-MNIST数据集, 然后将其加载到内存中"""
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在0到1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train,
                            batch_size,
                            shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,
                            batch_size,
                            shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_cifar10(batch_size,
                      train_transform=None,
                      test_transform=None,
                      root='../data'):
    """下载CIFAR10数据集, 然后将其加载到内存中"""
    if train_transform is None:
        train_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
    if test_transform is None:
        test_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
    cifar10_train = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=train_transform,
                                                 download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=root,
                                                train=False,
                                                transform=test_transform,
                                                download=True)
    return (data.DataLoader(cifar10_train,
                            batch_size,
                            shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(cifar10_test,
                            batch_size,
                            shuffle=False,
                            num_workers=get_dataloader_workers()))


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Animator:
    """在动画中绘制数据"""
    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1,
                 ncols=1,
                 figsize=(6, 4)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim,
                                            ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def cpu():
    return torch.device('cpu')


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


def train_gpu(net,
              train_iter,
              test_iter,
              loss,
              num_epochs,
              lr=0.1,
              optimizer=None,
              device=None):
    """用GPU训练模型"""
    if device is None:
        device = try_gpu()

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    animator = Animator(xlabel='epoch',
                        xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和, 训练准确率之和, 范例数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens"""
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]


class Vocab:
    """Vocabulary for text"""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(),
                                  key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(
            sorted(
                set(['<unk>'] + reserved_tokens + [
                    token
                    for token, freq in self.token_freqs if freq >= min_freq
                ])))
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


def load_corpus_time_machine(max_tokens=-1, token='word'):
    """返回时光机器数据集的词元索引列表和词汇表"""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    tokens = tokenize(lines, token)
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落,
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区, 随机范围包括`num_steps - 1`
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1, 是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # ⻓度为`num_steps`的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中,
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从`pos`位置开始的⻓度为`num_steps`的一个序列
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size

    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里, `initial_indices`包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - 1 - offset) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens,
                 token):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens, token)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size,
                           num_steps,
                           use_random_iter=False,
                           max_tokens=100000,
                           token='word'):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter,
                              max_tokens, token)
    return data_iter, data_iter.vocab


def get_rnn_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )  # tuple


def rnn(inputs, state, params):
    # `inputs`的形状:(`时间步数量`，`批量大小`，`词表大小`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # `X`的形状:(`批量大小`，`词表大小`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )


def get_gru_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(
            (num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    # `inputs`的形状:(`时间步数量`，`批量大小`，`词表大小`)
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # `X`的形状:(`批量大小`，`词表大小`)
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(
            (num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆状态参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
        W_hq, b_q
    ]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    # `inputs`的形状:(`时间步数量`，`批量大小`，`词表大小`)
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    # `X`的形状:(`批量大小`，`词表大小`)
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state,
                 forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


class RNNModule(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModule, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的, `num_directions`应该是2, 否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, X, state):
        # `X`的形状:(`批量大小`, `时间步数量`)
        # 我们将`X`转置为: (`时间步数量`, `批量大小`)
        X = F.one_hot(X.T.long(), self.vocab_size)
        # 经过OneHot处理之后为: (`时间步数量`, `批量大小`, `词表大小`), 再传给RNN层
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)

        # 全连接层首先将`Y`的形状改为(`时间步数`*`批量大小`, `隐藏单元数`)
        # 它的输出形状是 (`时间步数`*`批量大小`, `词表大小`)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size, device):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU`的隐藏状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # `nn.LSTM`的隐藏状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))


def predict_rnn(prefix, num_preds, net, vocab, device):
    """在`prefix`后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(
        (1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测期
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_rnn_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期"""
    state, timer = None, Timer()
    metric = Accumulator(2)  # 训练损失之和, 词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化`state`
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state`对于`nn.GRU`
                state.detach_()
            else:
                # `state`对于`nn.LSTM`或对于我们从零开始实现的模型
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)  # 注意: 这里要拉平, 因为输出结果也是拉平的
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了`mean`函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_rnn_gpu(net,
                  train_iter,
                  vocab,
                  lr,
                  num_epochs,
                  device=None,
                  use_random_iter=False,
                  predict_prefix=None):
    """训练模型"""
    if device is None:
        device = try_gpu()

    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch',
                        ylabel='perplexity',
                        legend=['train'],
                        xlim=[0, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_rnn(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_rnn_epoch(net, train_iter, loss, updater, device,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            if predict_prefix is not None:
                print(predict(predict_prefix))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')


def read_data_nmt():
    """载入'英语－法语'数据集"""
    data_dir = download_extract('fra_eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor(
        [truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=1000):
    """返回翻译数据集的迭代器和词汇表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source,
                      min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target,
                      min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


class Encoder(nn.Module):
    """编码器-解码器结构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """编码器-解码器结构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器结构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        # `pred` 的形状：(`batch_size`, `num_steps`, `vocab_size`)
        # `label` 的形状：(`batch_size`, `num_steps`)
        # `valid_len` 的形状：(`batch_size`,)
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        # `unweighted_loss`的形状: (`batch_size`, `num_steps`)
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        # `weighted_loss`的形状: (`batch_size`,)
        weighted_loss = (weights * unweighted_loss).mean(dim=1)
        return weighted_loss


def train_seq2seq_gpu(net, data_iter, lr, num_epochs, tgt_vocab, device=None):
    """训练序列到序列模型"""
    if device is None:
        device = try_gpu()

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
        elif type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_normal_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和, 词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat((bos, Y[:, :-1]), 1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行'反传'
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            if (epoch + 1) % 10 == 0:
                animator.add(epoch + 1, (metric[0] / metric[1], ))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def predict_seq2seq(net,
                    src_setence,
                    src_vocab,
                    tgt_vocab,
                    num_steps,
                    device,
                    save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将`net`设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_setence.lower().split(' ')] + [
        src_vocab['<eos>']
    ]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加batch_size轴
    # `enc_X`的形状是: (`1, `num_steps`)
    enc_X = torch.unsqueeze(torch.tensor(src_tokens,
                                         dtype=torch.long,
                                         device=device),
                            dim=0)
    # `enc_outputs[0]`的形状: (`num_steps`, `1`, `num_hiddens`)
    # `enc_outputs[1]`的形状: (`num_layers`, `1`, `num_hiddens`)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加batch_size轴
    # `dec_X`的形状是: (`1, `1`)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                         dtype=torch.long,
                                         device=device),
                            dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # `Y`的形状: (`1`, `1`, `vocab_size`)
        # `dec_state`的形状: (`num_layers`, `1`, `num_hiddens`)
        Y, dec_state = net.decoder(dec_X, dec_state)
        # `dec_X`的形状: (`1`, `1`)
        # 我们使用具有预测最高可能性的token, 作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测, 输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """计算 BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def annotate(text, xy, xytext):
    plt.gca().annotate(text, xy, xytext, arrowprops=dict(arrowstyle='->'))


def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]],
         'x',
         'f(x)',
         fmts=['-', '-o'])


def train_2d(trainer, steps=20, f_grad=None):
    """用定制的训练机优化2D目标函数"""
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


def show_trace_2d(results, f):
    """显示优化过程中2D变量的轨迹"""
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')


def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        """Defined in :numref:`sec_hybridize`"""
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')


class Residual(nn.Module):
    """The Residual block of ResNet"""
    def __init__(self,
                 input_channels,
                 num_channels,
                 use_1x1conv=False,
                 stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,
                               num_channels,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(num_channels,
                               num_channels,
                               kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,
                                   num_channels,
                                   kernel_size=1,
                                   stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels,
                 num_channels,
                 num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels,
                         num_channels,
                         use_1x1conv=True,
                         stride=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)


def resnet18(num_classes, in_channels=1):
    """一个简化版的ResNet-18 model"""
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU())
    net.add_module('resnet_block1', resnet_block(64, 64, 2, first_block=True))
    net.add_module('resnet_block2', resnet_block(64, 128, 2))
    net.add_module('resnet_block3', resnet_block(128, 256, 2))
    net.add_module('resnet_block4', resnet_block(256, 512, 2))
    net.add_module('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module('fc',
                   nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net


def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def bbox_to_rect(bbox, color):
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式:
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(xy=(bbox[0], bbox[1]),
                         width=bbox[2] - bbox[0],
                         height=bbox[3] - bbox[1],
                         fill=False,
                         edgecolor=color,
                         linewidth=2)


def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)  # 每个像素对应的box数量
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心, 需要设置偏移量
    # 因为一个像素的的高为1且宽为1, 我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # 生成锚框的所有中心点, 取值范围: (0, 1)
    # center_h.shape - (h, )
    # center_w.shape - (w, )
    # shift_x.shape - (h*w, )
    # shift_y.shape - (h*w, )
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成'boxes_per_pixel'个高和宽
    # 之后用于创建锚框的四角坐标
    # w.shape - (boxes_per_pixel, )
    # h.shape - (boxes_per_pixel, )
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                     sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                     * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 除以2来获得半高和半宽
    # anchor_manipulations.shape - (boxes_per_pixel*h*w, 4)
    anchor_manipulations = torch.stack(
        (-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有'boxes_per_pixel'个锚框，
    # 所以生成含所有锚框中心的网格, 重复了'boxes_per_pixel'次
    # out_grid.shape - (boxes_per_pixel*h*w, 4)
    #
    # 补充:
    # 这里要注意repeat()和repeat_interleave()在行为上的区别, 也就是在元素排列方式上的不同
    # 1. 将一个shape为(boxes_per_pixel, 4)的tensor通过repeat(h * w)
    #    变为(boxes_per_pixel*h*w, 4)
    # 2. 将一个shape为(h*w, 4)的tensor通过repeat_interleave(boxes_per_pixel)
    #    变为(boxes_per_pixel*h*w, 4)
    # 目的是为了让其对应逻辑元素对应, 之后可以相加
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)

    # output.shape - (boxes_per_pixel*h*w, 4)
    output = out_grid + anchor_manipulations

    # output.shape - (1, boxes_per_pixel*h*w, 4)
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (tuple, list)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0],
                      rect.xy[1],
                      labels[i],
                      va='center',
                      ha='center',
                      fontsize=9,
                      color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # `boxes1`, `boxes2`, `areas1`, `areas2`的形状:
    # `boxes1`: (boxes1的数量, 4)
    # `boxes2`: (boxes2的数量, 4)
    # `areas1`: (boxes1的数量,)
    # `areas2`: (boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # `inter_upperlefts`, `inter_lowerrights`, `inters`的形状:
    # (boxes1的数量, boxes2的数量, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # `inter_areas` and `union_areas`的形状:
    # (boxes1的数量, boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    # jaccard.shape - (num_anchors, num_gt_boxes)
    jaccard = box_iou(anchors, ground_truth)

    # 对于每个锚框, 分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors, ),
                                  -1,
                                  dtype=torch.long,
                                  device=device)
    # 根据阈值, 决定是否分配真实边界框
    # max_ious.shape - (num_anchors, )
    # indices.shape - (num_anchors, )
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors, ), -1)
    row_discard = torch.full((num_gt_boxes, ), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()  # 列索引
        anc_idx = (max_idx / num_gt_boxes).long()  # 行索引
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    # anchors_bbox_map.shape - (num_anchors, )
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    # c_anc.shape - (num_anchors, 4)
    # c_assigned_bb.shape - (num_anchors, 4)
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    # offset_xy.shape - (num_anchors, 2)
    # offset_wh.shape - (num_anchors, 2)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)

    # offset.shape - (num_anchors, 4)
    return offset


def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    # 参数:
    # anchors.shape - (batch_size, num_anchors, 4)
    # labels.shape - (batch_size, num_gt_boxes, 5)

    # 处理之后:
    # anchors.shape - (num_anchors, 4)
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        # label.shape - (num_gt_boxes, 5)
        label = labels[i, :, :]
        # anchors_bbox_map.shape(num_anchors, )
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # bbox_mask.shape(num_anchors, 4)
        # e.g.
        # tensor([[0., 0., 0., 0.],
        #         [1., 1., 1., 1.],
        #         [1., 1., 1., 1.],
        #         [0., 0., 0., 0.],
        #         [1., 1., 1., 1.]])
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将`类标签`和`分配的边界框坐标`初始化为零
        class_labels = torch.zeros(num_anchors,
                                   dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4),
                                  dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别
        # 如果一个锚框没有被分配, 我们标记其为背景(值为零)
        # indices_true.shape - (正类锚框的数量, 1)
        # bb_idx.shape - (正类锚框的数量, 1)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        # class_labels.shape - (num_anchors, )
        # assigned_bb.shape - (num_anchors, 4)
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        # 偏移量转换
        # offset.shape - (num_anchors, 4)
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))  # (num_enchors*4, )
        batch_mask.append(bbox_mask.reshape(-1))  # (num_echors*4, )
        batch_class_labels.append(class_labels)  # (num_anchors, )

    bbox_offset = torch.stack(batch_offset)  # (batch_size, num_enchors*4)
    bbox_mask = torch.stack(batch_mask)  # (batch_size, num_enchors*4)
    class_labels = torch.stack(batch_class_labels)  # (batch_size, num_enchors)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    # anc.shape - (num_anchors, 4)
    anc = box_corner_to_center(anchors)

    # pred_bbox_xy.shape - (num_anchors, 2)
    # pred_bbox_wh.shape - (num_anchors, 2)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]

    # pred_bbox.shape - (num_anchors, 4)
    # predicted_bbox.shape - (num_anchors, 4)
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


# 目录结构(图片的尺寸都是256x256):
# banana_detection/
#   bananas_train/
#     images/
#       0.png
#       1.png
#       2.png
#       ...
#     label.csv
#   bananas_val/
#     images/
#       0.png
#       1.png
#       2.png
#       ...
#     label.csv
#
# label.csv格式:
# img_name,label,xmin,ymin,xmax,ymax
# 0.png,0,104,20,143,58
# 1.png,0,68,175,118,223
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = download_extract('banana_detection')
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'images', f'{img_name}')))
        # label,xmin,ymin,xmax,ymax
        targets.append(list(target))

    # 图片尺寸的大小是256x256, 最终除以256表示坐标在图片的相对位置
    # image是一个长度为1000的list, 每个元素image.shape: (3, 256, 256)
    # targets.shape: (1000, 1, 5)
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) +
              (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size,
                                             shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter


def load_data_airfoil_self_noise(batch_size=10, n=1500):
    data = np.genfromtxt(download('airfoil_self_noise'),
                         dtype=np.float32,
                         delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = load_array((data[:n, :-1], data[:n, -1]),
                           batch_size,
                           is_train=True)
    return data_iter, data.shape[1] - 1


def train_optimization(trainer_fn,
                       states,
                       hyperparams,
                       data_iter,
                       feature_dim,
                       num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0,
                     std=0.01,
                     size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: linreg(X, w, b), squared_loss
    # Train
    animator = Animator(xlabel='epoch',
                        ylabel='loss',
                        xlim=[0, num_epochs],
                        ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()  # 计算loss的均值
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter),
                             (evaluate_loss(net, data_iter, loss), ))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]


def train_concise_optimization(trainer_fn,
                               hyperparams,
                               data_iter,
                               feature_dim,
                               num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(feature_dim, 1))

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)

    loss = nn.MSELoss()
    animator = Animator(xlabel='epoch',
                        ylabel='loss',
                        xlim=[0, num_epochs],
                        ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y) / 2  # PyTorch的MSE Loss实现和我们的不同
            l.backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter),
                             (evaluate_loss(net, data_iter, loss) / 2, ))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]


def show_heatmaps(matrices,
                  xlabel,
                  ylabel,
                  titles=None,
                  figsize=(3.5, 3.5),
                  cmap='Reds'):
    """
    matrices的形状是: (要显示的行数, 要显示的列数, 查询的数目, 键的数目)
    """
    use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows,
                             num_cols,
                             figsize=figsize,
                             sharex=True,
                             sharey=True,
                             squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
