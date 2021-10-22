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
from foxrelax.go.gotypes import Player
from foxrelax.go.utils import (print_board, print_move, clear_screen)


@click.group()
def cli():
    pass


@cli.command()
def run():
    board_size = 9
    game = goboard_slow.GameState.new_game(board_size)
    bots = {Player.BLACK: RandomBot(), Player.WHITE: RandomBot()}
    while not game.is_over():
        time.sleep(0.3)
        clear_screen()
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)


if __name__ == "__main__":
    cli()