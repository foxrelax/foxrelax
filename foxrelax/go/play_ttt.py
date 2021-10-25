#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import time
import click

from foxrelax.go.ttt.ttttype import (Player, Point)

COL_NAMES = 'ABC'

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def print_board(board):
    print('   A   B   C')
    for row in (1, 2, 3):
        pieces = []
        for col in (1, 2, 3):
            piece = board.get(Point(row, col))
            if piece == Player.X:
                pieces.append('X')
            elif piece == Player.O:
                pieces.append('O')
            else:
                pieces.append(' ')
        print('%d  %s' % (row, ' | '.join(pieces)))


def point_from_coords(text):
    col_name = text[0]
    row = int(text[1])
    return Point(row, COL_NAMES.index(col_name) + 1)


@click.group()
def cli():
    pass


@cli.command()
def run():
    pass


if __name__ == "__main__":
    cli()