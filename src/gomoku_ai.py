"""
A stronger Gomoku AI engine with board heuristics and alpha-beta search.
Enhanced: stronger pattern score, transposition table, full human vs AI game
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

EMPTY = 0
BLACK = 1
WHITE = 2

# 四个搜索方向
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# 棋型分数: (连子数, 空端数)
# 补充更多棋型分数，强化攻防能力
PATTERN_SCORES = {
    (5, 0): 1_000_000_000,   # 五连
    (4, 0): 100_000_000,     # 活四
    (4, 1): 50_000_000,      # 冲四 - 提高权重，优先防守/进攻
    (3, 0): 10_000_000,      # 活三 - 提高权重
    (3, 1): 1_000_000,       # 眠三 - 提高权重
    (2, 0): 100_000,         # 活二
    (2, 1): 10_000,          # 眠二
    (1, 0): 1_000,           # 单子
}

@dataclass(frozen=True)
class Move:
    row: int
    col: int
    player: int


class GomokuBoard:
    def __init__(self, size: int = 15) -> None:
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.moves: List[Move] = []
        self.count = 0

    def copy(self) -> GomokuBoard:
        new_board = GomokuBoard(self.size)
        new_board.grid = [row.copy() for row in self.grid]
        new_board.moves = self.moves.copy()
        new_board.count = self.count  # 修复 bug
        return new_board

    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == EMPTY

    def play(self, row: int, col: int, player: int) -> None:
        if not self.is_valid(row, col):
            raise ValueError(f"Invalid move: ({row},{col})")
        self.grid[row][col] = player
        self.moves.append(Move(row, col, player))
        self.count += 1

    def undo(self) -> None:
        if not self.moves:
            raise ValueError("No moves to undo")
        last = self.moves.pop()
        self.grid[last.row][last.col] = EMPTY
        self.count -= 1

    def is_full(self) -> bool:
        return self.count >= self.size * self.size

    def winner(self) -> Optional[int]:
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] != EMPTY and self.is_five_in_a_row(row, col):
                    return self.grid[row][col]
        return None

    def is_five_in_a_row(self, row: int, col: int) -> bool:
        player = self.grid[row][col]
        for dr, dc in DIRECTIONS:
            length = 1
            # 正反两个方向延伸
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == player:
                    length += 1
                    r += sign * dr
                    c += sign * dc
            if length >= 5:
                return True
        return False

    def get_neighbors(self, distance: int = 2) -> List[Tuple[int, int]]:
        neighbors = set()
        for move in self.moves:
            for dr in range(-distance, distance + 1):
                for dc in range(-distance, distance + 1):
                    r = move.row + dr
                    c = move.col + dc
                    if 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == EMPTY:
                        neighbors.add((r, c))
        # 棋盘空盘，默认落中心
        if not neighbors:
            center = self.size // 2
            return [(center, center)]
        return list(neighbors)

    def print_board(self) -> None:
        header = '   ' + ' '.join(f'{i:2}' for i in range(self.size))
        print(header)
        for i, row in enumerate(self.grid):
            line = f'{i:2} '
            for cell in row:
                if cell == EMPTY:
                    line += '.  '
                elif cell == BLACK:
                    line += 'X  '
                else:
                    line += 'O  '
            print(line)


class GomokuAI:
    def __init__(self, size: int = 15, time_limit: float = 1.5) -> None:
        self.size = size
        self.time_limit = time_limit
        self.start_time = 0.0
        self.transposition: dict[Tuple[Tuple[int, ...], int], int] = {}

    def best_move(self, board: GomokuBoard, player: int) -> Tuple[int, int]:
        self.start_time = time.time()
        self.transposition.clear()  # 每步清空置换表，避免旧缓存干扰
        best_move = None
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf
        depth = self._select_depth(board)

        ordered_moves = self._generate_moves(board, player)
        for move in ordered_moves:
            board.play(move[0], move[1], player)
            score = self._alpha_beta(board, depth - 1, alpha, beta, self._opponent(player))
            board.undo()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
            if self._timed_out():
                break

        if best_move is None:
            raise RuntimeError("No move found")
        return best_move

    def _timed_out(self) -> bool:
        return time.time() - self.start_time > self.time_limit

    def _select_depth(self, board: GomokuBoard) -> int:
        # 开局浅搜索，中局、残局加深搜索，平衡速度与棋力
        total_cells = self.size * self.size
        if board.count < 8:
            return 4
        elif board.count < 35:
            return 6
        elif board.count < total_cells // 2:
            return 7
        else:
            return 5

    def _generate_moves(self, board: GomokuBoard, player: int) -> List[Tuple[int, int]]:
        candidates = board.get_neighbors(distance=2)
        scored_moves = []
        for row, col in candidates:
            score = self._move_heuristic(board, row, col, player)
            scored_moves.append(((row, col), score))
        # 按启发分降序排序，优先搜索高分点
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        return [move for move, _ in scored_moves[:min(25, len(scored_moves))]]

    def _move_heuristic(self, board: GomokuBoard, row: int, col: int, player: int) -> int:
        """评估一个移动的启发值，包括赢棋检测和防守检测"""
        opponent = self._opponent(player)
        
        # 1. 检查是否能直接赢棋
        board.grid[row][col] = player
        if board.is_five_in_a_row(row, col):
            board.grid[row][col] = EMPTY
            return 1_000_000_000  # 赢棋，最高优先级
        board.grid[row][col] = EMPTY
        
        # 2. 检查对方是否能在此位置赢棋（必须防守）
        board.grid[row][col] = opponent
        if board.is_five_in_a_row(row, col):
            board.grid[row][col] = EMPTY
            return 900_000_000  # 防守对方赢棋，次高优先级
        board.grid[row][col] = EMPTY
        
        # 3. 评估该位置对自己的得分
        board.grid[row][col] = player
        own_score = self._evaluate_position(board, row, col, player)
        board.grid[row][col] = EMPTY
        
        # 4. 评估该位置对对方的得分（防守价值）
        board.grid[row][col] = opponent
        opp_score = self._evaluate_position(board, row, col, opponent)
        board.grid[row][col] = EMPTY
        
        # 综合得分：自己的得分 + 防守对方的权重（防守优先级更高）
        return own_score + opp_score * 1.5

    def _alpha_beta(self, board: GomokuBoard, depth: int, alpha: float, beta: float, player: int):
        if self._timed_out():
            return 0

        winner = board.winner()
        if winner == player:
            return math.inf
        if winner == self._opponent(player):
            return -math.inf
        if depth == 0 or board.is_full():
            return self._evaluate(board, player)

        board_key = self._hash_board(board, player)
        if board_key in self.transposition:
            return self.transposition[board_key]

        moves = self._generate_moves(board, player)
        if not moves:
            return self._evaluate(board, player)

        value = -math.inf
        for row, col in moves:
            board.play(row, col, player)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, self._opponent(player))
            board.undo()

            if score > value:
                value = score
            alpha = max(alpha, value)
            if alpha >= beta:
                break
            if self._timed_out():
                break

        self.transposition[board_key] = value
        return value

    def _evaluate(self, board: GomokuBoard, player: int) -> int:
        own_score = self._evaluate_player(board, player)
        opp_score = self._evaluate_player(board, self._opponent(player))
        return own_score - opp_score

    def _evaluate_player(self, board: GomokuBoard, player: int) -> int:
        total = 0
        for row in range(board.size):
            for col in range(board.size):
                if board.grid[row][col] == player:
                    total += self._evaluate_position(board, row, col, player)
        return total

    def _evaluate_position(self, board: GomokuBoard, row: int, col: int, player: int) -> int:
        score = 0
        for dr, dc in DIRECTIONS:
            line = self._build_line(board, row, col, dr, dc, player)
            score += self._score_line(line)
        return score

    def _build_line(self, board: GomokuBoard, row: int, col: int, dr: int, dc: int, player: int):
        window = [player]
        # 向两个方向拓展棋子窗口
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < board.size and 0 <= c < board.size and len(window) < 10:
                window.append(board.grid[r][c])
                r += sign * dr
                c += sign * dc
        return window

    def _score_line(self, line: Sequence[int]) -> int:
        player = line[0]
        if player == EMPTY:
            return 0
        best = 0
        n = len(line)
        for start in range(n):
            if line[start] != player:
                continue
            count = 0
            idx = start
            while idx < n and line[idx] == player:
                count += 1
                idx += 1
            # 判断两端是否为空
            left_open = (start - 1 >= 0) and (line[start - 1] == EMPTY)
            right_open = (idx < n) and (line[idx] == EMPTY)
            pattern_score = self._pattern_score(count, left_open, right_open)
            if pattern_score > best:
                best = pattern_score
        return best

    def _pattern_score(self, count: int, left_open: bool, right_open: bool) -> int:
        if count >= 5:
            return PATTERN_SCORES[(5, 0)]
        open_ends = int(left_open) + int(right_open)
        key = (count, open_ends)
        return PATTERN_SCORES.get(key, 0)

    def _hash_board(self, board: GomokuBoard, player: int) -> Tuple[Tuple[int, ...], int]:
        flat = tuple(cell for row in board.grid for cell in row)
        return flat, player

    @staticmethod
    def _opponent(player: int) -> int:
        return BLACK if player == WHITE else WHITE


# 完整人机对战游戏入口
def play_gomoku_game():
    board = GomokuBoard(size=15)
    ai = GomokuAI(size=15, time_limit=1.8)

    print("=" * 50)
    print("       🎮 强化版五子棋 AI 游戏 🎮")
    print("=" * 50)
    print("规则：你是黑方(X)，AI是白方(O)")
    print("落子输入格式：行 列 （例如：7 7 代表棋盘中心）")
    print("输入 'q' 可退出游戏")
    print("=" * 50)

    move_count = 0
    while True:
        board.print_board()
        print()
        
        winner = board.winner()
        if winner == BLACK:
            print("🎉 恭喜！你战胜了AI！")
            break
        if winner == WHITE:
            print("😥 AI获胜，再接再厉！")
            break
        if board.is_full():
            print("⚖️ 棋盘已满，平局！")
            break

        # 玩家落子
        while True:
            try:
                in_str = input("\n请输入落子坐标 (或 'q' 退出)：").strip()
                if in_str.lower() == 'q':
                    print("感谢游玩！再见！")
                    return
                x, y = map(int, in_str.split())
                if board.is_valid(x, y):
                    board.play(x, y, BLACK)
                    move_count += 1
                    print(f"你落子：{x} {y}")
                    break
                else:
                    print("❌ 坐标无效，请选择空白位置！")
            except ValueError:
                print("❌ 输入格式错误，请输入两个数字，空格分隔！")

        # 玩家落子后判断胜负
        if board.winner() or board.is_full():
            continue

        # AI 落子
        print("\n🤖 AI 思考中...")
        ai_row, ai_col = ai.best_move(board, WHITE)
        board.play(ai_row, ai_col, WHITE)
        move_count += 1
        print(f"AI 落子：{ai_row} {ai_col}")
        print()


def run_example() -> None:
    """运行示例：展示AI在特定局面的推荐走法"""
    print("\n" + "=" * 50)
    print("       📊 AI 走法推荐示例 📊")
    print("=" * 50)
    board = GomokuBoard(size=15)
    ai = GomokuAI(size=15, time_limit=1.5)
    
    starting_moves = [
        (7, 7, BLACK),
        (7, 8, WHITE),
        (8, 7, BLACK),
        (8, 8, WHITE),
    ]
    
    print("初始局面：")
    for row, col, player in starting_moves:
        board.play(row, col, player)
    
    board.print_board()
    
    print("\n🤖 AI (黑方) 思考中...")
    move = ai.best_move(board, BLACK)
    print(f"AI 推荐走法：{move}\n")


# 主菜单
def main_menu():
    while True:
        print("\n" + "=" * 50)
        print("         五子棋 AI 主菜单")
        print("=" * 50)
        print("1. 人机对战")
        print("2. 查看 AI 走法示例")
        print("3. 退出")
        print("=" * 50)
        
        choice = input("请选择 (1/2/3)：").strip()
        
        if choice == '1':
            play_gomoku_game()
        elif choice == '2':
            run_example()
        elif choice == '3':
            print("感谢使用！再见！")
            break
        else:
            print("❌ 输入错误，请重新选择！")


if __name__ == "__main__":
    main_menu()
