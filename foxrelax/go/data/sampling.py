# -*- coding:utf-8 -*-
import os
import random
from foxrelax.go.data.index_processor import KGSIndex
"""
采样模块
1. 确保随机选择指定数量的棋局
2. 确保训练样本和测试样本必须是分开的, 不能出现任何重叠

样本的格式如下:
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


class Sampler:
    def __init__(self,
                 data_dir='../data',
                 num_test_games=200,
                 cap_year=2018,
                 seed=1337):
        self.data_dir = data_dir
        self.num_test_games = num_test_games
        self.test_games = []  # 测试样本
        self.train_games = []  # 训练样本
        self.test_samples = 'test_samples.cache'  # 测试样本的缓存文件
        self.cap_year = cap_year

        random.seed(seed)
        self.compute_test_samples()

    def draw_samples(self, num_sample_games):
        """
        从index中采样, 数量为num_sample_games

        注意: 只会采样满足year <= cap_year的样本

        >>> sampler.draw_sample(10)
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

        # 从index中读取所有的games
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()
        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            # 大于cap_year的直接跳过
            if year > self.cap_year:
                continue
            num_games = fileinfo['num_games']
            for i in range(num_games):
                available_games.append((filename, i))
        print('>>> Total number of games used: ' + str(len(available_games)))

        # 采样: 每次采样一个样本, 直到满足数量为止
        sample_set = set()
        while len(sample_set) < num_sample_games:
            sample = random.choice(available_games)
            if sample not in sample_set:
                sample_set.add(sample)
        print('Drawn ' + str(num_sample_games) + ' samples:')
        return list(sample_set)

    def compute_test_samples(self):
        """
        计算测试样本

        1. 如果不存在self.test_samples, 创建一个本地文件存储固定的测试样本
        2. 如果self.test_samples存在, 则直接读取其中的内容作为测试样本
        3. 将测试样本写入self.test_games
        """
        test_samples_path = os.path.join(self.data_dir, self.test_samples)
        if not os.path.isfile(test_samples_path):
            test_games = self.draw_samples(self.num_test_games)
            test_samples_file = open(test_samples_path, 'w')
            for sample in test_games:
                test_samples_file.write(str(sample) + "\n")
            test_samples_file.close()

        test_samples_file = open(test_samples_path, 'r')
        sample_contents = test_samples_file.read()
        test_samples_file.close()
        for line in sample_contents.split('\n'):
            if line != "":
                (filename, index) = eval(line)
                self.test_games.append((filename, index))

    def draw_training_games(self):
        """
        将训练样本写入self.train_games(会过滤掉self.test_games中的测试样本)
        """
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()
        for file_info in index.file_info:
            filename = file_info['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            num_games = file_info['num_games']
            for i in range(num_games):
                sample = (filename, i)
                if sample not in self.test_games:  # 过滤掉test game
                    self.train_games.append(sample)
        print('total num training games: ' + str(len(self.train_games)))

    def draw_training_samples(self, num_sample_games):
        """
        采样(会过滤掉测试样本), 返回num_sample_games个训练样本
        """
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()
        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            num_games = fileinfo['num_games']
            for i in range(num_games):
                available_games.append((filename, i))
        print('total num games: ' + str(len(available_games)))

        # 采样: 每次采样一个样本, 直到满足数量为止
        sample_set = set()
        while len(sample_set) < num_sample_games:
            sample = random.choice(available_games)
            if sample not in self.test_games:  # 过滤掉test game
                sample_set.add(sample)
        print('Drawn ' + str(num_sample_games) + ' samples:')
        return list(sample_set)

    def draw_all_training(self):
        """
        采样(会过滤掉测试样本), 返回所有训练样本
        """
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()
        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            if 'num_games' in fileinfo.keys():
                num_games = fileinfo['num_games']
            else:
                continue
            for i in range(num_games):
                available_games.append((filename, i))
        print('total num games: ' + str(len(available_games)))

        sample_set = set()
        for sample in available_games:
            if sample not in self.test_games:
                sample_set.add(sample)
        print('Drawn all samples, ie ' + str(len(sample_set)) + ' samples:')
        return list(sample_set)

    def draw_data(self, data_type, num_samples):
        if data_type == 'test':
            if num_samples < len(self.test_games):
                return self.test_games[:num_samples]
            else:
                return self.test_games
        elif data_type == 'train' and num_samples is not None:
            return self.draw_training_samples(num_samples)
        elif data_type == 'train' and num_samples is None:
            return self.draw_all_training()
        else:
            raise ValueError(
                data_type +
                " is not a valid data type, choose from 'train' or 'test'")