# -*- coding:utf-8 -*-
import os
import shutil
import hashlib
import tarfile
import zipfile
import requests
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
from .data_hub import DATA_HUB


def _mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


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


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    Plot a list of images

    imgs需要(H, W, C)或者(H, W)这样的格式
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def download_data_cats_and_dogs_small(base_dir='../data/cats_and_dogs_small',
                                      download_fold=None):
    # cats_and_dogs_small/
    #     train/
    #         cats/
    #         dogs/
    #     validation/
    #         cats/
    #         dogs/
    #     test/
    #         cats/
    #         dogs/
    original_dataset_dir = download_extract('kaggle_cats_and_dogs',
                                            download_fold)
    _mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    _mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    _mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    _mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')
    _mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    _mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    _mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    _mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, 'cats')
    _mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    _mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
    return base_dir
