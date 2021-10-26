# -*- coding:utf-8 -*-
import enum
import random

from foxrelax.go.agent.base import Agent

__all__ = [
    'MinimaxAgent',
]


class GameResult(enum.Enum):
    LOSS = 1
    DRAW = 2
    WIN = 3


def reverse_game_result(game_result):
    if game_result == GameResult.LOSS:
        return game_result.WIN
    if game_result == GameResult.WIN:
        return game_result.LOSS
    return GameResult.DRAW


def best_result(game_state):
    """
    对于game_state.next_player来说最好的结果
    """
    if game_state.is_over():
        # 游戏结束
        if game_state.winner() == game_state.next_player:
            return GameResult.WIN
        elif game_state.winner() is None:
            return GameResult.DRAW
        else:
            return GameResult.LOSS
    # 首先循环遍历所有可能动作, 并计算下一个游戏状态, 接着假设`对方会尽力反击的假象动作`.
    # 对这个新棋局调用best_result, 得到对方从这个新棋局所能够获得的最佳结果, 这个结果的
    # 反面就是己方结果. 最后在遍历完所有动作之后, 选择能给己方带来最佳结果的那个动作
    best_result_so_far = GameResult.LOSS
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)
        # 找到对方的最佳动作
        opponent_best_result = best_result(next_state)  # 递归调用
        our_result = reverse_game_result(opponent_best_result)
        if our_result.value > best_result_so_far.value:
            best_result_so_far = our_result
    return best_result_so_far


class MiniMaxAgent(Agent):
    def select_move(self, game_state):
        winning_moves = []
        draw_moves = []
        losing_moves = []
        for possible_move in game_state.legal_moves():
            # 计算如果选择这个动作, 会导致什么游戏状态
            next_state = game_state.apply_move(possible_move)
            # 由于下一回合对方执子, 因此需要找到对方可能获得的最佳结果,
            # 这锅结果的反面就是己方结果
            opponent_best_outcome = best_result(next_state)
            our_best_outcome = reverse_game_result(opponent_best_outcome)
            if our_best_outcome == GameResult.WIN:
                winning_moves.append(possible_move)
            elif our_best_outcome == GameResult.DRAW:
                draw_moves.append(possible_move)
            else:
                losing_moves.append(possible_move)
        if winning_moves:
            return random.choice(winning_moves)
        if draw_moves:
            return random.choice(draw_moves)
        return random.choice(losing_moves)
