# -*- coding:utf-8 -*-
import sys
import os
import tarfile
import gzip
import shutil
import glob
import multiprocessing
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.go.encoder.base import get_encoder_by_name
from foxrelax.go.data.index_processor import KGSIndex
from foxrelax.go.data.generator import DataGenerator
from foxrelax.go.data.sampling import Sampler
from foxrelax.go.gosgf import Sgf_game
from foxrelax.go.goboard_fast import (Board, GameState, Move)
from foxrelax.go.gotypes import (Player, Point)


def _worker(jobinfo):
    try:
        clazz, encoder, zip_file, data_file_name, game_list = jobinfo
        clazz(encoder=encoder).process_zip(zip_file, data_file_name, game_list)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


class GoDataProcessor:
    def __init__(self, encoder='oneplane', data_directory='../data'):
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory

    def load_go_data(self,
                     data_type='train',
                     num_samples=1000,
                     use_generator=False):
        """
        从KGS下载在线围棋棋谱, 加载, 处理, 存储数据
    
        data_type: train | test
        num_samples: 棋局的数量
        """

        # 从KGS下载所有的棋局数据, 并存储在本地, 如果数据已经存在, 则不会重新下载
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()

        # 采样
        sampler = Sampler(data_dir=self.data_dir)
        # data:
        # [('KGS-2004-19-12106-.tar.gz', 7883),
        #  ('KGS-2006-19-10388-.tar.gz', 10064),
        #  ('KGS-2012-19-13665-.tar.gz', 8488),
        #  ...
        #  ('KGS-2009-19-18837-.tar.gz', 1993),
        #  ('KGS-2005-19-13941-.tar.gz', 9562),
        #  ('KGS-2003-19-7582-.tar.gz', 265),
        #  ('KGS-2009-19-18837-.tar.gz', 9086),
        #  ('KGS-2005-19-13941-.tar.gz', 13444)]
        data = sampler.draw_data(data_type, num_samples)

        self.map_to_workers(data_type, data)
        if use_generator:
            generator = DataGenerator(self.data_dir, data)
            return generator
        else:
            features_and_labels = self.consolidate_games(data_type, data)
            return features_and_labels

    def map_to_workers(self, data_type, samples):
        """
        并行的处理samepls文件

        参数:
        data_type: train | test
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
        zip_names = set()  # 保存samples中所有的文件名*.tar.gz
        # indices_by_zip_name:
        # {
        #   'KGS-2013-19-13783-.tar.gz': [5696, 3746, 8279, ... ],
        #   'KGS-2008-19-14002-.tar.gz': [10428, 7795, 1509, ...],
        #   ...
        # }
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        # 需要并行处理的*.tar.gz文件
        zips_to_process = []
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir + '/' + data_file_name):
                zips_to_process.append(
                    (self.__class__, self.encoder_string, zip_name,
                     data_file_name, indices_by_zip_name[zip_name]))

        cores = multiprocessing.cpu_count()
        # Bug Fix:
        # ValueError: not enough values to unpack
        if cores > len(zip_names):
            cores = len(zip_names)
        pool = multiprocessing.Pool(processes=cores)
        p = pool.map_async(_worker, zips_to_process)
        try:
            _ = p.get()
        except KeyboardInterrupt:  # Caught keyboard interrupt, terminating workers
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def process_zip(self, zip_file_name, data_file_name, game_list):
        """
        一个*.tar.gz文件中会有多个*.sgf文件, 我们要处理的是game_list中列出来的*.sgf文件

        参数格式:
        zip_file_name: KGS-2013-19-13783-.tar.gz, 包含多个*.sgf的压缩文件
        data_file_name: KGS-2013-19-13783-train, 是生成的目标文件的前缀
        game_list: [5696, 3746, 8279, ... ], 需要处理的*.sgf索引列表

        内部实现逻辑:
        1. 调用unzip_data解压当前文件
        2. 初始化一个Encoder实例来编码SGF棋谱(直接使用self.encoder)
        3. 初始化合理形状的特征和标签NumPy数组
        4. 迭代遍历棋局列表, 并逐个处理棋局数据
           a. 每一局开始前处理让子的逻辑self.get_handicap
           b. 将每一回合的下一步动作编码为label
           c. 将每一回合的当前棋盘布局状态编码为feature
           d. 把下一步动作执行到棋盘上并继续
        5. 在本地文件系统分块存储特征和标签

        之所以要分块存储, 因为NumPy数组会迅速增大, 而数据存储在较小文件中可以保留更多灵活性, 
        例如: 我们可以把分块文件合并起来, 也可以根据需要将每个文件单独加载到内存. 我们在实现
        while循环中分块的最后一部分数据可能会丢掉, 但是影响不大

        最终会生成:
        KGS-2013-19-13783-train_features_0.npy
        KGS-2013-19-13783-train_features_1.npy
        KGS-2013-19-13783-train_features_2.npy
        ...
        KGS-2013-19-13783-train_labels_0.npy
        KGS-2013-19-13783-train_labels_1.npy
        KGS-2013-19-13783-train_labels_2.npy
        ...
        """
        print(f'process zip: {zip_file_name}')
        # 调用unzip_data解压当前文件
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        # name_list:
        # ['kgs-19-2013',
        #  'kgs-19-2013/2013-07-30-6.sgf',
        #  'kgs-19-2013/2013-03-20-12.sgf',
        #  ...
        # ]
        name_list = zip_file.getnames()
        total_examples = self.num_total_examples(zip_file, game_list,
                                                 name_list)
        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        # 初始化合理形状的特征和标签NumPy数组
        # 如果: total_examples = 634
        # 则: shape = (1, 19, 19), feature_shape = [634  1  19  19]
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples, ))

        # 迭代遍历棋局(*.sgf)列表, 并逐个处理棋局数据
        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content)
            # 每一局开始前处理让子的逻辑
            game_state, first_move_done = self.get_handicap(sgf)

            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                    # 编码一个样本
                    # 将每一回合的下一步动作编码为label
                    # 将每一回合的当前棋盘布局状态编码为feature
                    if first_move_done and point is not None:
                        features[counter] = self.encoder.encode(game_state)
                        labels[counter] = self.encoder.encode_point(point)
                        counter += 1
                    # 把下一步动作执行到棋盘上并继续
                    game_state = game_state.apply_move(move)
                    first_move_done = True

        # 在本地文件系统分块存储特征和标签, 以1024为一个块
        feature_file_base = self.data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = self.data_dir + '/' + data_file_name + '_labels_%d'

        chunk = 0
        chunksize = 1024
        while features.shape[0] >= chunksize:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[
                chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)

    def unzip_data(self, zip_file_name):
        """
        将*.tar.gz解压成*.tar
        
        例如:
        zip_file_name: KGS-2013-19-13783-.tar.gz -> KGS-2013-19-13783-.tar
        """
        this_gz = gzip.open(self.data_dir + '/' +
                            zip_file_name)  # 将`gz`文件解压成`tar`文件

        tar_file = zip_file_name[0:-3]  # 删除文件名结尾的*.gz
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)  # 将解压后的内容写入到`tar`文件
        this_tar.close()
        return tar_file

    @staticmethod
    def get_handicap(sgf):
        """
        获取让子数, 并将它们布置在空白棋盘上
        例如:
        HA[5]...AB[dd][pd][jj][dp][pp]

        sgf.get_handicap() - HA(让子)
        sgf.get_root().get_setup_stones() - AB(Add Black), AW(Add White), AE(Add Empty)
        """
        go_board = Board(19, 19)
        first_move_done = False  # 如果有让子, 设置为True
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.BLACK, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.WHITE, None, move)
        return game_state, first_move_done

    def num_total_examples(self, zip_file, game_list, name_list):
        """
        计算压缩文件中有效动作总数

        game_list: [5696, 3746, 8279, ... ]
        name_list: ['kgs-19-2013',
                    'kgs-19-2013/2013-07-30-6.sgf',
                    'kgs-19-2013/2013-03-20-12.sgf',
                    ...]
        name_list是文件名的列表, game_list是要统计的索引

        注意: 对于有让子的棋局, 第一个动作就可以作为样本; 对于没有让子的棋局, 从第二个动作开始才能作为样本

        >>> tar_file = process.unzip_data('KGS-2013-19-13783-.tar.gz')
        >>> zip_file = tarfile.open(process.data_dir + '/' + tar_file)
        >>> process.num_total_examples(zip_file, game_list=[0, 1],
                                       name_list=[
                                         'kgs-19-2013',
                                         'kgs-19-2013/2013-07-30-6.sgf',
                                         'kgs-19-2013/2013-03-20-12.sgf'])
        218
        """
        total_examples = 0
        # 遍历所有的棋局*.sgf
        for index in game_list:
            name = name_list[index + 1]  #跳过第一个, 这个文件不是*.sgf
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                _, first_move_done = self.get_handicap(sgf)

                num_moves = 0  # 一个*.sgf中一共有多少move
                # 遍历一局
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        # 对于有让子的棋局, 第一个动作就可以作为样本; 对于没有让子的棋局, 从第二个动作开始才能作为样本
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples

    def consolidate_games(self, data_type, samples):
        """
        将分块的数据合并成一个目标文件:
        features_train.npy
        labels_train.npy
        或者:
        features_test.npy
        labels_test.npy

        使用oneplane encoder, 当num_samples=1000的情况下(有1000个sgf文件), 样本大约是233M, 标签大约是1.3M. 
        这样推算如果有10000个*.sgf文件, 样本大约是2GB, 标签大约是13M

        参数说明:
        data_type: train | test
        samples: [('KGS-2004-19-12106-.tar.gz', 7883),
                  ('KGS-2006-19-10388-.tar.gz', 10064),
                  ('KGS-2012-19-13665-.tar.gz', 8488),
                  ...
                  ('KGS-2009-19-18837-.tar.gz', 1993),
                  ('KGS-2005-19-13941-.tar.gz', 9562),
                  ('KGS-2003-19-7582-.tar.gz', 265),
                  ('KGS-2009-19-18837-.tar.gz', 9086),
                  ('KGS-2005-19-13941-.tar.gz', 13444)]
        """
        files_needed = set(file_name for file_name, index in samples)
        # file_names:
        # ['KGS-2004-19-12106-',
        #  'KGS-2006-19-10388-',
        #  'KGS-2012-19-13665-',
        #  ...]
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + data_type
            file_names.append(file_name)

        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            print(f'consolidate_game: {file_prefix}')
            base = self.data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                feature_list.append(x)
                label_list.append(y)
        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        # 保存为文件
        np.save('{}/features_{}.npy'.format(self.data_dir, data_type),
                features)
        np.save('{}/labels_{}.npy'.format(self.data_dir, data_type), labels)
        # 返回
        return features, labels


if __name__ == "__main__":
    process = GoDataProcessor()
    generator = process.load_go_data(use_generator=True)
    print(generator.get_num_samples())  # 168960
    for X, y in generator._generate():
        print(X.shape, y.shape)  # (128, 1, 19, 19) (128,)
        break