# -*- coding:utf-8 -*-
import torch
import glob
from typing import Generator
import numpy as np


class DataGenerator:
    def __init__(self, data_directory, samples):
        """
        samples:
        [('KGS-2004-19-12106-.tar.gz', 7883),
         ('KGS-2006-19-10388-.tar.gz', 10064),
         ('KGS-2012-19-13665-.tar.gz', 8488),
         ...
         ('KGS-2009-19-18837-.tar.gz', 1993),
         ('KGS-2005-19-13941-.tar.gz', 9562),
         ('KGS-2003-19-7582-.tar.gz', 265),
         ('KGS-2009-19-18837-.tar.gz', 9086),
         ('KGS-2005-19-13941-.tar.gz', 13444)]
        """
        self.data_directory = data_directory
        self.samples = samples
        self.files = set(file_name for file_name, _ in samples)
        self.num_samples = None

    def get_num_samples(self, batch_size=128):
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size):
                self.num_samples += X.shape[0]
            return self.num_samples

    def _generate(self, batch_size):
        """
        创建并返回批量数据

        功能和GoDataProcessor.consolidate_games()类似. 不同的是前者会把数据加载到内存
        一次性返回(需要一个巨大的NumPy数组); 而_generate只需要yield一个小批量数据即可
        """
        # 遍历所有的棋局*.sgf
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = self.data_directory + '/' + file_name + '_features_*.npy'
            # 遍历一个棋局内所有的训练样本
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                while x.shape[0] >= batch_size:
                    x_batch, x = x[:batch_size], x[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    yield torch.tensor(x_batch), torch.tensor(y_batch).type(
                        torch.long)

    def generate(self, batch_size=128):
        """
        这个会无限循环下去
        """
        while True:
            print('xxxx')
            for item in self._generate(batch_size):
                yield item


if __name__ == '__main__':
    samples = [('KGS-2004-19-12106-.tar.gz', 7883),
               ('KGS-2006-19-10388-.tar.gz', 10064),
               ('KGS-2012-19-13665-.tar.gz', 8488),
               ('KGS-2009-19-18837-.tar.gz', 1993),
               ('KGS-2005-19-13941-.tar.gz', 9562),
               ('KGS-2003-19-7582-.tar.gz', 265),
               ('KGS-2009-19-18837-.tar.gz', 9086),
               ('KGS-2005-19-13941-.tar.gz', 13444)]
    generator = DataGenerator(data_directory='../data', samples=samples)
    for X, y in generator._generate():
        print(X.shape,
              y.shape)  # torch.Size([128, 1, 19, 19]) torch.Size([128])
        break
    print(generator.get_num_samples())
