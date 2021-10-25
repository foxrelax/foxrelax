# -*- coding:utf-8 -*-
from collections import namedtuple
from foxrelax.go.gotypes import (Player, Point)
"""
在自己的回合中, 双方都可以选择不落子, 从而跳过当前回合. 如果双方接连跳过回合, 比赛就结束了

在计算评分时, `死棋`的处理方式与吃子完全相同.

围棋比赛的目标是`比对方控制棋盘上更大的底盘`. 计算得分有两种不同的方法, 但是得出的结果通常是相同的

1. 数目法: 棋盘上每一颗被己方棋子完全包围的交叉点都记做一分, 称为`一目`. 己方吃掉的每颗棋子也被算作一分,
   加起来谁的总分高谁就获胜
2. 数子法: 在数子法中, 每一目算一分, 己方在棋盘上剩下的每颗棋子也算一分

除及特殊情况外, 这两种方法得到的胜负结果一般是相同的. 如果没有过早结束棋局的话, 双方提子数的差别与双方棋盘上
剩下棋子的差别往往是一样的. 数目法在休闲棋局中更常见, 但对计算机而言, 数子法更方便

此外, 执白子的一方还要得到额外的分数, 以补偿后手的劣势. 这种补偿称为`贴子`. 在数目法中一般贴6.5子, 在数子法中
一般贴7.5子. 这里额外的0.5子用来确保不会出现平局 
"""


class Territory:
    """
    一个`territory_map`将棋盘分为: `tones`, `territory`和`dame`三种类型
    """
    def __init__(self, territory_map) -> None:
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.BLACK:
                self.num_black_stones += 1
            elif status == Player.WHITE:
                self.num_white_stones += 1
            elif status == 'territory_b':
                self.num_black_territory += 1
            elif status == 'territory_w':
                self.num_white_territory += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)


class GameResult(namedtuple('GameResult', 'b w komi')):
    @property
    def winner(self):
        if self.b > self.w + self.komi:
            return Player.BLACK
        return Player.WHITE

    @property
    def winning_margin(self):
        w = self.w + self.komi
        return abs(self.b - w)

    def __str__(self):
        w = self.w + self.komi
        if self.b > w:
            return 'B+%.1f' % (self.b - w, )
        return 'W+%.1f' % (w - self.b, )


def evaluate_territory(board):
    """
    将board映射成: `territory`和`dame`

    1. 如果point已经访问过了则直接跳过
    2. 如果point已经有棋子了, 直接写入到status
    3. 如果point完全被一种颜色包围(它的边界只有一种颜色), 可以划定为`territory`
    4. 如果point没有被一种颜色包围, 则这个point属于`dame`
    """
    status = {}
    for r in range(1, board.num_rows + 1):
        for c in range(1, board.num_cols + 1):
            p = Point(row=r, col=c)
            # 如果point已经访问过了则直接跳过
            if p in status:
                continue
            stone = board.get(p)
            # 如果point已经有棋子了, 直接写入到status
            if stone is not None:
                status[p] = stone
            else:
                group, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:
                    # 如果point完全被一种颜色包围(它的边界只有一种颜色), 可以划定为`territory`
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.BLACK else 'w'
                    fill_with = f'territory_{stone_str}'
                else:
                    # 如果point没有被一种颜色包围, 则这个point属于`dame`
                    fill_with = 'dame'
                for pos in group:
                    status[pos] = fill_with
    return Territory(status)


def _collect_region(start_pos, board, visited=None):
    """
    从start_pos搜寻棋盘上连续的部分, 以及这个部分所有的边界
    """
    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()

    all_points = [start_pos]  # 和自己同色的point
    all_borders = set()  # 边界的颜色, 可能有0,1,2个元素
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta_r, delta_c in deltas:
        next_p = Point(row=start_pos.row + delta_r,
                       col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            points, borders = _collect_region(next_p, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add(neighbor)
    return all_points, all_borders


def compute_game_result(game_state):
    territory = evaluate_territory(game_state.board)
    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        komi=7.5)
