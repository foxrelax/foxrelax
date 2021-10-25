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
from foxrelax.go.gotypes import Player
from foxrelax.go.utils import (print_board, print_move, clear_screen)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--board',
              default='slow',
              type=click.Choice(['slow', 'normal', 'fast']),
              help='board模式')
def run(board):
    pass


if __name__ == "__main__":
    cli()