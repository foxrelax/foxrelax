# -*- coding:utf-8 -*-
from enum import Enum, unique
from collections import namedtuple

__all__ = ['Player', 'Point']


@unique
class Player(Enum):
    BLACK = 1
    WHITE = 2

    @property
    def other(self):
        return Player.BLACK if self == Player.WHITE else Player.WHITE


class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

    def __deepcopy__(self, memodict={}):
        """
        Point一旦创建, 是不可变的
        """
        return self