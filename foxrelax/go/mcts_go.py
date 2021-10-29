#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import click

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from foxrelax.go.utils import (print_board, print_move, point_from_coords)
from foxrelax.go.goboard_fast import (GameState, Move)
from foxrelax.go.gotypes import Player
from foxrelax.go.mcts.mcts import MCTSAgent

BOARD_SIZE = 5


@click.group()
def cli():
    pass


@cli.command()
def run():
    game = GameState.new_game(BOARD_SIZE)
    bot = MCTSAgent(10000, 1.4)

    while not game.is_over():
        print_board(game.board)
        if game.next_player == Player.BLACK:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == "__main__":
    cli()