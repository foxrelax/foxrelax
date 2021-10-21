# -*- coding:utf-8 -*-

__all__ = ['Agent']


class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        """
        所有的机器人都要根据当前的游戏状态选择一个动作
        """
        raise NotImplemented()

    def diagnostics(self):
        return {}