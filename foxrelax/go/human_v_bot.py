#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import time
import click

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from foxrelax.go.agent.naive import RandomBot
from foxrelax.go import goboard_slow
from foxrelax.go import goboard
from foxrelax.go import goboard_fast
from foxrelax.go.gotypes import Player
from foxrelax.go.utils import (print_board, print_move, clear_screen,
                               point_from_coords)
"""
人机对战(机器人是随机机器人)
"""


@click.group()
def cli():
    pass


@cli.command()
@click.option('--board',
              default='slow',
              type=click.Choice(['slow', 'normal', 'fast']),
              help='board模式')
def run(board):
    board_size = 9
    if board == 'slow':
        game = goboard_slow.GameState.new_game(board_size)
    elif board == 'normal':
        game = goboard.GameState.new_game(board_size)
    elif board == 'fast':
        game = goboard_fast.GameState.new_game(board_size)
    # 构造一个随机机器人
    bot = RandomBot()
    while not game.is_over():
        clear_screen()
        print_board(game.board)
        if game.next_player == Player.BLACK:
            human_move = input('--')
            point = point_from_coords(human_move)
            move = goboard.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == "__main__":
    cli()