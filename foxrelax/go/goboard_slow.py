# -*- coding:utf-8 -*-

import copy
from foxrelax.go.gotypes import (Point, Player)

__all__ = ['Board', 'GameState', 'Move']


class IllegalMoveError(Exception):
    pass


class Move:
    """
    表示一个回合中可能采取的动作. 有三种动作:
    1. 在棋盘上落下一颗棋子(play)
    2. 跳过回合(pass)
    3. 直接认输(resign)

    遵循美国围棋协会(AGA)的惯例, 我们使用术语动作(move)来表示这三种行动中的一个. 在实际棋局中, 需要传递一个Point对象指定落子的位置.
    在使用中我们通常调用Move.play(), Move.pass_turn(), Move.resign()来构造一个动作, 而不是直接调用Move的构造函数
    """
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        return Move(point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)


class GoString:
    """
    棋链(string of stones), 将同色相连的棋子组合成一个整体, 同时跟踪这个整体的状态以及它们的气

    1. 可以调用num_liberties来获取任意交叉点的处的气数
    2. remove_liberty, 当对方在这条棋链相邻的地方落子时, 棋链的气通常会减少
    3. add_liberty, 当这条棋链或者己方其它棋链吃掉对方棋子的时候, 棋链的气会增加
    4. 当一次落子把两颗棋子连接起来的时候, 需要调用merge_with方法

    使用set()可以高效的实现这个结构, 方便的计算集合的交集, 并集, 差集等
    """
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)  # 棋子的Point
        self.liberties = set(liberties)  # 气的Point

    def remove_liberty(self, point):
        self.liberties.remove(point)

    def add_liberty(self, point):
        self.liberties.add(point)

    def merge_with(self, go_string):
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color, combined_stones,
            (self.liberties | go_string.liberties - combined_stones))

    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties


class Board:
    """
    棋盘

    最主要的功能是可以`落子`, 落子后自动更新棋盘的状态
    """
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}  # 用来跟踪棋盘的状态, 是一个棋链的字典

    def place_stone(self, player, point):
        """
        落子

        1. 检测point是否在棋盘内以及point是否已经有棋子
        2. 合并任何同色且相邻的棋链
        3. 减少对方所有相邻棋链的气
        4. 如果对方某条棋链没有气了, 要提走它
        """
        # 检测point是否在棋盘内以及point是否已经有棋子
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None

        adjacent_same_color = []  # point周围同色棋链
        adjacent_opposite_color = []  # point周围对方的棋链
        liberties = []  # point周围的气
        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
        new_string = GoString(player, [point], liberties)

        # 合并任何同色且相邻的棋链
        for same_color_string in adjacent_same_color:
            new_string = new_string.merge_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        # 减少对方所有相邻棋链的气
        for other_color_string in adjacent_opposite_color:
            other_color_string.remove_liberty(point)

        # 如果对方某条棋链没有气了, 要提走它
        for other_color_string in adjacent_opposite_color:
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string)

    def is_on_grid(self, point):
        """
        检查point是否在棋盘内
        """
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def get(self, point):
        """
        返回point位置的color
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point):
        """
        返回point位置的棋链
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def _remove_string(self, string):
        """
        提走一条棋链的所有棋子

        在提走一条棋链的时候, 其它棋链会增加气数
        """
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    neighbor_string.add_liberty(point)
            self._grid[point] = None

    def __eq__(self, other):
        return isinstance(other, Board) and \
            self.num_rows == other.num_rows and \
            self.num_cols == other.num_cols and \
            self._grid == other._grid


class GameState:
    """
    游戏状态
    """
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous  # 这样整个游戏状态就构成了一条链
        self.last_move = move

    def apply_move(self, move):
        """
        执行move动作之后, 返回新的GameState对象
        """
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.BLACK, None, None)

    def is_over(self):
        """
        判断棋局是否结束

        1. 如果一方直接认输(resign)则棋局结束
        2. 如果双方接连跳过回合(pass)则棋局结束
        """
        if self.last_move is None:
            return False

        # 如果一方直接认输(resign)则棋局结束
        if self.last_move.is_resign:
            return True

        # 如果双方接连跳过回合(pass)则棋局结束
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def is_move_self_capture(self, player, move):
        """
        判断是否是`自吃(self capture)`

        当棋链只剩下一口气时, 如果在这口气所对应的Point上落子, 导致己方气尽而被提走, 我们称为`自吃`. 大多数围棋规则都禁止这样的
        动作(也有例外, 四年一度的应氏杯世界职业围棋锦标赛Ing Cup就允许这种自吃的落子规则, 它是国际围棋比赛中奖项最大的比赛之一).
        我们的代码中是`禁止自吃的`, 这样做和现行规则保持一致, 同时也减少机器人需要考虑的动作数目. 要构造出一个选择自吃是最佳解决方案
        的场景也不是不可能, 但在正式比赛中这种情况基本上闻所未闻

        在检查新落棋子是否气尽之前, 一般应当先判断是否能吃掉对方的棋子, 在所有规则中, 这一步动作都是有效的吃子而不是自吃

        Board类实际上是允许自吃动作的, 但是在GameState类中, 我们可以在Board的一个副本上执行落子动作, 然后检查副本上的气数并检查这个规则 
        """
        if not move.is_play:
            return False

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0

    @property
    def situation(self):
        """
        游戏状态包括: 棋盘上所有棋子的位置, 以及下一回合的执子方
        """
        return (self.next_player, self.board)

    def does_move_violate_ko(self, player, move):
        """
        判断是否是`劫争(ko)`

        为了保证棋局最终能够结束, 那些会让棋局回到之前某个状态的落子动作是禁止的. 在实现劫争规则的时候要注意`反提(snapback)`. 劫争
        规则一般归纳为`不能立即重新吃掉对方棋子`

        我们的实现方式如下: 棋手的一次落子不能让棋局恢复到上一回合的游戏状态. 这里, 游戏状态包括`棋盘上所有棋子的位置, 以及下一回合的执子方`,
        这种组合称为situational superko rule. 由于每个GameState实例都存储了指向前一状态的指针, 我们可以从当前状态一直往回遍历所有的历史状态,
        # 并对比是否有劫争情况发生

        这个本的实现逻辑简单, 但是速度比较慢, 每一次落子, 都必须创建一个棋局游戏状态的deepcopy, 并将它与之前所有的历史状态进行对比, 历史状态的
        数量会随着时间推移增加
        """
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board)
        past_state = self.previous_state
        while past_state is not None:
            if past_state.situation == next_situation:
                return True
            past_state = past_state.previous_state
        return False

    def is_valid_move(self, move):
        """
        判断动作是否合法

        1. 确认落子的Point是空的(Board.place_stone会做这个检测)
        2. 检查落子是自吃
        3. 检查落子不违反劫争规则
        """
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (self.board.get(move.point) is None
                and not self.is_move_self_capture(self.next_player, move)
                and not self.does_move_violate_ko(self.next_player, move))

    def legal_moves(self):
        """
        返回当前合法的moves
        """
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row=row, col=col))
                if self.is_valid_move(move):
                    moves.append(move)
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    def winner(self):
        # TODO
        return None
