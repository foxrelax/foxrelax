# -*- coding:utf-8 -*-

__all__ = ['Agent']


class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplemented()

    def diagnostics(self):
        return {}