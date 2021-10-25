# -*- coding:utf-8 -*-
from enum import Enum, unique
from collections import namedtuple

__all__ = ['Player', 'Point']


@unique
class Player(Enum):
    X = 1
    O = 2

    @property
    def other(self):
        return Player.X if self == Player.O else Player.O


class Point(namedtuple('Point', 'row col')):
    def __deepcopy__(self, memodict={}):
        """
        Point一旦创建, 是不可变的, 直接返回自己
        """
        return self