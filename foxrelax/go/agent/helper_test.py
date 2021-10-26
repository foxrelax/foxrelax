#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from foxrelax.go.gotypes import (Player, Point)
from foxrelax.go.goboard import Board
from foxrelax.go.agent.helper import is_point_an_eye


class EyeTest(unittest.TestCase):
    def test_corner(self):
        board = Board(19, 19)
        #  5  .  .  .  .  .
        #  4  .  .  .  .  .
        #  3  .  .  .  .  .
        #  2  x  x  .  .  .
        #  1  .  x  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.BLACK, Point(1, 2))
        board.place_stone(Player.BLACK, Point(2, 2))
        board.place_stone(Player.BLACK, Point(2, 1))
        self.assertTrue(is_point_an_eye(board, Point(1, 1), Player.BLACK))
        self.assertFalse(is_point_an_eye(board, Point(1, 1), Player.WHITE))

    def test_corner_false_eye(self):
        board = Board(19, 19)
        #  5  .  .  .  .  .
        #  4  .  .  .  .  .
        #  3  .  .  .  .  .
        #  2  x  .  .  .  .
        #  1  .  x  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.BLACK, Point(1, 2))
        board.place_stone(Player.BLACK, Point(2, 1))
        self.assertFalse(is_point_an_eye(board, Point(1, 1), Player.BLACK))

        #  5  .  .  .  .  .
        #  4  .  .  .  .  .
        #  3  .  .  .  .  .
        #  2  x  o  .  .  .
        #  1  .  x  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.WHITE, Point(2, 2))
        self.assertFalse(is_point_an_eye(board, Point(1, 1), Player.BLACK))

    def test_middle(self):
        board = Board(19, 19)
        #  5  .  .  .  .  .
        #  4  .  x  x  x  .
        #  3  .  x  .  x  .
        #  2  .  x  x  x  .
        #  1  .  .  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.BLACK, Point(2, 2))
        board.place_stone(Player.BLACK, Point(3, 2))
        board.place_stone(Player.BLACK, Point(4, 2))
        board.place_stone(Player.BLACK, Point(4, 3))
        board.place_stone(Player.WHITE, Point(4, 4))
        board.place_stone(Player.BLACK, Point(3, 4))
        board.place_stone(Player.BLACK, Point(2, 4))
        board.place_stone(Player.BLACK, Point(2, 3))
        self.assertTrue(is_point_an_eye(board, Point(3, 3), Player.BLACK))


if __name__ == '__main__':
    unittest.main()
