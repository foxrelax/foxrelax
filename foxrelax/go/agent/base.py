# -*- coding:utf-8 -*-

__all__ = ['Agent']


class Agent:
    """
    机器人代理, 代理只需要一个核心功能, 告诉它当前的game_state, 
    它可以返回当前应该做的动作
    """
    def __init__(self):
        pass

    def select_move(self, game_state):
        """
        所有的机器人都要根据当前的游戏状态选择一个动作
        """
        raise NotImplemented()

    def diagnostics(self):
        return {}