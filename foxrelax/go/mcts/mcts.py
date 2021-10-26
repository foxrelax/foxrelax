# -*- coding:utf-8 -*-
import math
import random

from foxrelax.go.agent.base import Agent
from foxrelax.go.gotypes import Player
from foxrelax.go.utils import coords_from_point

__all__ = [
    'MCTSAgent',
]


def fmt(x):
    # 格式化Player
    if x is Player.BLACK:
        return 'B'
    if x is Player.WHITE:
        return 'W'
    # 格式化Move
    if x.is_pass:
        return 'pass'
    if x.is_resign:
        return 'resign'
    return coords_from_point(x.point)
