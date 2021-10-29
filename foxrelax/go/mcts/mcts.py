# -*- coding:utf-8 -*-
import math
import random

from foxrelax.go.agent.base import Agent
from foxrelax.go.agent.naive_fast import FastRandomBot
from foxrelax.go.gotypes import Player
from foxrelax.go.utils import coords_from_point

__all__ = [
    'MCTSAgent',
]
"""
蒙特卡洛树搜索(MCTS)为我们提供了一种方法, 可以在不依赖任何围棋策略知识的前提下评估游戏状态. MCTS
算法不需要利用游戏特有的启发式规则, 而是通过模拟随机棋局来评估棋局的好坏. 我们把模拟进行的每一个随机
棋局称为一次推演(rollout)或者拟盘(playout)

我们让两个随机AI相互对抗, 如果这时候发现黑方持续比白方赢得多, 那么一定是黑方从一开始就掌握了某种优势.
因此要评估某个棋局的状态, 我们可以从那里开始进行多次随机对弈推演, 来弄清楚这个棋局是否对某一方有利, 而且
也不需要理解为何这个棋局是有利的

当然也可能遇到不平衡的结果, 如果模拟10次随机对弈, 白方赢了7次而黑方赢了3次. 我们是否有信心说白方有优势呢?
实际上并不能, 因此推演的次数太少了. 如果白方在100次随机对弈中获胜了70次, 我们就可以几乎确定初始棋局对白方
有利, 因此, 关键点在于更多次的推演

MCTS算法一般包含三个步骤:
<1> 将新的棋局添加到MCTS树种
<2> 从这个新添加的棋局开始模拟随机博弈
<3> 根据随机博弈的结果更新树节点的统计信息

构建的搜索树每个节点都记录了该节点之后任意棋局开始的胜负计数, 也就是说节点计数包括其所有子节点的总和
"""


def fmt(x):
    # 格式化Player
    if x is Player.BLACK:
        return 'B'
    if x is Player.WHITE:
        return 'W'
    # 格式化Move
    if x.is_pass:
        return 'pass'
    if x.is_resign:
        return 'resign'
    return coords_from_point(x.point)


def show_tree(node, indent='', max_depth=3):
    if max_depth < 0:
        return
    if node is None:
        return
    if node.parent is None:
        print('%sroot' % indent)
    else:
        player = node.parent.game_state.next_player
        move = node.move
        print('%s%s %s %d %.3f' % (
            indent,
            fmt(player),
            fmt(move),
            node.num_rollouts,
            node.winning_frac(player),
        ))
    for child in sorted(node.children,
                        key=lambda n: n.num_rollouts,
                        reverse=True):
        show_tree(child, indent + '  ', max_depth - 1)


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state  # 搜索树当前节点的游戏状态
        self.parent = parent  # 当前MCTSNode节点的父节点(要表示树的根节点, 可以把它的parent设置为None)
        self.move = move  # 触发当前棋局的上一步动作

        # 保存推演的统计信息
        self.win_counts = {
            Player.BLACK: 0,
            Player.WHITE: 0,
        }
        self.num_rollouts = 0  # 总的推演数量

        self.children = []  # 当前节点所有的子节点列表
        # 从当前棋局开始, 所有可能的合法动作列表, 这个列表只记录那些还没有成为树中某一节点的动作, 每当搜索树添加
        # 一个新节点时, 我们都会从unvisited_moves中提取一个动作, 为它生成一个新的MCTSNode, 并添加到children中
        self.unvisited_moves = game_state.legal_moves()

    def add_random_child(self):
        """
        向树中添加新节点
        """
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        """
        更当前节点的统计信息
        """
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        """
        当前节点中是否还有合法动作尚未添加到树中
        """
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        """
        检测是否到了终盘
        """
        return self.game_state.is_over()

    def winning_frac(self, player):
        """
        返回某一方在推演中获胜的比例
        """
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        """
        用当前的游戏状态作为根节点来创建一颗新的搜索树, 接着反复生成新的推演. 我们每一回合使用固定
        轮数的推演, 有的其它实现按照固定时长

        <1> 每一轮开始推演, 先沿着搜索树往下遍历, 直至找到一个可以添加子节点的节点(任何还留有尚未
        添加到树中的合法动作的棋局)为止. select_child负责挑选可供继续搜索的最佳分支
        <2> 找到合适的节点后, 调用add_random_child来选择一个后续动作, 并将它添加到搜索树, 此时的
        node是一个新创建的MCTSNode, 它还没有包含任何推演
        <3> 调用simulate_random_game模拟玩一局游戏
        <4> 最后需要为新创建的节点以及它的所有祖先节点更新获胜统计信息
        """
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            # <1> 每一轮开始推演, 先沿着搜索树往下遍历, 直至找到一个可以添加子节点的节点(任何还留有尚未
            # 添加到树中的合法动作的棋局)为止. select_child负责挑选可供继续搜索的最佳分支
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # <2> 找到合适的节点后, 调用add_random_child来选择一个后续动作, 并将它添加到搜索树, 此时的
            # node是一个新创建的MCTSNode, 它还没有包含任何推演
            if node.can_add_child():
                node = node.add_random_child()

            # <3> 调用simulate_random_game模拟玩一局游戏
            winner = self.simulate_random_game(node.game_state)

            # <4> 最后需要为新创建的节点以及它的所有祖先节点更新获胜统计信息
            while node is not None:
                node.record_win(winner)
                node = node.parent

        # 打印得分前10的10的10个move
        scored_moves = [(child.winning_frac(game_state.next_player),
                         child.move, child.num_rollouts)
                        for child in root.children]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:10]:
            print('%s - %.3f (%d)' % (m, s, n))

        # 遍历搜索树顶端的所有分支, 选择获胜率最高的那个分支
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move

    def select_child(self, node):
        """
        从node.children中选出UCT Score最大的child返回, 如果node.children为空则返回None

        UCT Score原理:
        每一回合AI能够利用的时间是有限的, 这意味着能够执行的推演次数也是有限的. 每多一次推演都能提高
        一个可能动作的评估水平. 我们可以把推演看做某种总量有限的资源, 如果在动作A上多消耗一次推演, 在
        动作B上就得少消耗一次推演, 我们需要一个策略来决定如何分配有限的预算. 标准的策略称为`搜索树置信
        区间上界公式(upper confidence bound for trees formula, UCT)`, UCT可以在下面两个相互冲
        突的目标之间取得平衡:
        目标一: 花费更多的时间去检查最佳动作, 我们把这个目标称为`深入挖掘(exploitation)`. 即想要对至今
        为止搜索到的比较理想的目标深入挖掘. 这么做需要花费更多的时间来对那些估计获胜率最高的动作进行推演,
        你的估计就会更加准确. 误报率将会随着推演数量的增多而下降

        目标二: 如果某个节点只被访问了几次, 那么得到的评估可能有很大的偏差, 即使纯属偶然也可能遇到一个实际
        很好的动作得出估计获胜率很低的情况. 这时候如果能多几次推演, 就可能发觉出它真实的价值. 因此这个目标
        就是花费更多的时间来提高那些被访问次数最少的分支的评估准确率, 这个目标我们称为`广泛探索(exploration)`

        深入挖掘和广泛探索之间的权衡, 是很多试错型算法的共同特征

        计算公式:
        uct_score = w + c x sqrt{logN/n}
        <1> w - 节点的胜率(win_percentage)
        <2> c - 表示权衡`深入挖掘`和`广泛探索的权重`(temperature) 
        <3> N - 表示推演总数(total_rollouts)
        <4> n - 从当前节点开始所有的推演数

        w表示节点的胜率; 被访问次数越少的节点, sqrt{logN/n}就越大; c在w和sqrt{logN/n}之间做权衡. 如果使用
        较大的c, 就会花费更多的时间去访问那些访问次数最少的节点; 而使用较小的c就会花费更多的时间来对最有希望的
        节点进行更准确的评估. 合适的c往往经过反复试错才能找到. 通常从1.5开始. 参数c有时候被称为温度(temperature),
        当温度高的时候, 搜索更加发散; 当温度低的时候, 搜索更加集中
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # 遍历所有的child
        for child in node.children:
            # 计算每一个child的UCT Score
            win_percentage = child.winning_frac(node.game_state.next_player)
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            # 找出UCT Score最大的child
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.BLACK: FastRandomBot(),
            Player.WHITE: FastRandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()