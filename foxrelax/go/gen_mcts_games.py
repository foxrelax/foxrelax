#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import random
import click
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from foxrelax.go.encoder.base import get_encoder_by_name
from foxrelax.go import goboard_fast as goboard
from foxrelax.go.mcts.mcts import MCTSAgent
from foxrelax.go.utils import (print_board, print_move)


def generate_game(board_size, rounds, max_moves, temperature):
    # boards存放编码后的棋盘状态
    # moves存放编码后的落子动作
    boards, moves = [], []

    # 用给定的棋盘尺寸, 按名称初始化一个OnePlaneEncoder
    encoder = get_encoder_by_name('oneplane', board_size)

    # 一个尺寸为board_size的新棋局实例被初始化好了
    game = goboard.GameState.new_game(board_size)

    # 指定推演回合和温度参数, 生成一个蒙特卡洛树搜索代理作为机器人
    bot = MCTSAgent(rounds, temperature)

    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        # 机器人选择下一步动作
        move = bot.select_move(game)
        if move.is_play:
            # 把编码的棋盘状态添加到boards数组中
            boards.append(encoder.encode(game))

            # 把下一动作进行one hot encode编码, 并添加到moves数组中
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)

        # 把机器人的下一步动作执行到棋盘上
        print_move(game.next_player, move)
        game = game.apply_move(move)
        num_moves += 1
        if num_moves > max_moves:
            break

    return np.array(boards), np.array(moves)


@click.group()
def cli():
    pass

    # parser.add_argument('--board-out')
    # parser.add_argument('--move-out')


@cli.command()
@click.option('--board-size', default=9, type=int, help='board size')
@click.option('--rounds', default=1000, type=int, help='mcts rounds')
@click.option('--temperature',
              default=0.8,
              type=float,
              help='mcts temperature')
@click.option('--max-moves', default=60, type=int, help='max moves per game')
@click.option('--num-games', default=10, type=int, help='number of games')
@click.option('--board-out', default=None, type=str, help='board out')
@click.option('--move-out', default=None, type=str, help='move out')
def run(board_size, rounds, temperature, max_moves, num_games, board_out,
        move_out):
    if board_out is None:
        print('please input --board-out')
        return
    if move_out is None:
        print('please input --move-out')
        return

    xs = []
    ys = []

    for i in range(num_games):
        print('Generating game %d/%d...' % (i + 1, num_games))
        x, y = generate_game(board_size, rounds, max_moves, temperature)
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    np.save(board_out, x)
    np.save(move_out, y)


"""
执行: ./gen_mcts_games.py run --board-out=features.npy --move-out=labels.npy
"""
if __name__ == "__main__":
    cli()