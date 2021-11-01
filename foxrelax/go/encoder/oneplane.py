# -*- coding:utf-8 -*-
import numpy as np

from foxrelax.go.encoder.base import Encoder
from foxrelax.go.goboard import Point


class OnePlaneEncoder(Encoder):
    """
    用1表示下一回合的执子方; -1表示其对手(而不是用固定的1表示黑方, -1表示白方).

    当前的实现的版本有一个特征平面, 还有多个特征平面的实现方式(一个平面表示黑方, 一个平面
    表示白方, 一个平面表示劫争)
    """
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    def name(self):
        return 'oneplane'

    def encode(self, game_state):
        """
        对于棋盘上的每一个交叉点
        <1> 如果该点落下的是当前执子方, 则在矩阵中填充1(自己填充1)
        <2> 如果是对手方的棋子, 则填充-1(对手填充-1)
        <3> 如果该点为空, 则填充0
        """
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    continue
                if go_string.color == next_player:
                    board_matrix[0, r, c] = 1
                else:
                    board_matrix[0, r, c] = -1
        return board_matrix

    def encode_point(self, point):
        """
        将围棋棋盘上的交叉点(point)转换为整数索引

        point -> index
        """
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        """
        将整数索引转换为围棋棋盘上的交叉点(point)

        index -> point
        """
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return OnePlaneEncoder(board_size)