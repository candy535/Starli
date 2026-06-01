"""
An advanced Gomoku AI engine with Zobrist Hashing, Killer Moves, and History Heuristic.
Optimizations: Zobrist Hash, Persistent Transposition Table, Killer Moves, History Heuristic.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

EMPTY = 0
BLACK = 1
WHITE = 2

# 四个搜索方向
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# 棋型分数
PATTERN_SCORES = {
    (5, 0): 1_000_000_000,   # 五连
    (4, 0): 100_000_000,     # 活四
    (4, 1): 50_000_000,      # 冲四
    (3, 0): 10_000_000,      # 活三
    (3, 1): 1_000_000,       # 眠三
    (2, 0): 100_000,         # 活二
    (2, 1): 10_000,          # 眠二
    (1, 0): 1_000,           # 单子
}

# Zobrist 哈希表初始化
random.seed(42)
ZOBRIST_TABLE = [
    [[random.getrandbits(64) for _ in range(3)] for _ in range(15)]
    for _ in range(15)
]
ZOBRIST_BLACK_TURN = random.getrandbits(64)


@dataclass(frozen=True)
class Move:
    row: int
    col: int
    player: int


class ZobristHasher:
    """Zobrist 哈希计算器"""
    
    def __init__(self):
        self.hash = 0
    
    def put_piece(self, row: int, col: int, player: int) -> None:
        """放置棋子"""
        self.hash ^= ZOBRIST_TABLE[row][col][player]
    
    def remove_piece(self, row: int, col: int, player: int) -> None:
        """移除棋子"""
        self.hash ^= ZOBRIST_TABLE[row][col][player]
    
    def get_hash(self) -> int:
        return self.hash
    
    def copy(self) -> ZobristHasher:
        """复制哈希器"""
        hasher = ZobristHasher()
        hasher.hash = self.hash
        return hasher


class GomokuBoard:
    def __init__(self, size: int = 15) -> None:
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.moves: List[Move] = []
        self.count = 0
        self.zobrist = ZobristHasher()

    def copy(self) -> GomokuBoard:
        new_board = GomokuBoard(self.size)
        new_board.grid = [row.copy() for row in self.grid]
        new_board.moves = self.moves.copy()
        new_board.count = self.count
        new_board.zobrist = self.zobrist.copy()
        return new_board

    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == EMPTY

    def play(self, row: int, col: int, player: int) -> None:
        if not self.is_valid(row, col):
            raise ValueError(f"Invalid move: ({row},{col})")
        self.grid[row][col] = player
        self.moves.append(Move(row, col, player))
        self.count += 1
        self.zobrist.put_piece(row, col, player)

    def undo(self) -> None:
        if not self.moves:
            raise ValueError("No moves to undo")
        last = self.moves.pop()
        self.grid[last.row][last.col] = EMPTY
        self.count -= 1
        self.zobrist.remove_piece(last.row, last.col, last.player)

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


@dataclass
class TranspositionEntry:
    """置换表条目"""
    depth: int
    value: int
    flag: str  # 'exact', 'lower', 'upper'
    timestamp: int = 0


class GomokuAI:
    def __init__(self, size: int = 15, time_limit: float = 2.0) -> None:
        self.size = size
        self.time_limit = time_limit
        self.start_time = 0.0
        
        # 持久置换表（不每步清空）
        self.transposition: Dict[int, TranspositionEntry] = {}
        self.transposition_hits = 0
        self.transposition_misses = 0
        
        # 杀手走法表：killer_moves[depth] = [(row1, col1), (row2, col2)]
        self.killer_moves: Dict[int, List[Tuple[int, int]]] = {}
        
        # 历史启发表：history[player][(row, col)] = score
        self.history: Dict[int, Dict[Tuple[int, int], int]] = {
            BLACK: {},
            WHITE: {}
        }
        
        self.search_count = 0
        self.timestamp = 0

    def best_move(self, board: GomokuBoard, player: int) -> Tuple[int, int]:
        self.start_time = time.time()
        self.search_count = 0
        self.timestamp += 1
        self.killer_moves.clear()
        
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
            if self._timed_out():
                break

        if best_move is None:
            raise RuntimeError("No move found")
        
        # 统计信息
        hit_rate = self.transposition_hits / max(1, self.transposition_hits + self.transposition_misses)
        print(f"[DEBUG] 搜索节点数: {self.search_count}, 置换表命中率: {hit_rate:.2%}, 表大小: {len(self.transposition)}")
        
        return best_move

    def _timed_out(self) -> bool:
        return time.time() - self.start_time > self.time_limit

    def _select_depth(self, board: GomokuBoard) -> int:
        total_cells = self.size * self.size
        if board.count < 8:
            return 5
        elif board.count < 35:
            return 7
        elif board.count < total_cells // 2:
            return 8
        else:
            return 6

    def _generate_moves(self, board: GomokuBoard, player: int) -> List[Tuple[int, int]]:
        """生成并排序走法，结合杀手走法、历史启发"""
        candidates = board.get_neighbors(distance=2)
        scored_moves = []
        
        for row, col in candidates:
            score = self._move_heuristic(board, row, col, player)
            scored_moves.append(((row, col), score))
        
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        return [move for move, _ in scored_moves[:min(25, len(scored_moves))]]

    def _move_heuristic(self, board: GomokuBoard, row: int, col: int, player: int) -> int:
        """评估走法，包含防守检测"""
        opponent = self._opponent(player)
        
        # 检查赢棋
        board.grid[row][col] = player
        if board.is_five_in_a_row(row, col):
            board.grid[row][col] = EMPTY
            return 1_000_000_000
        board.grid[row][col] = EMPTY
        
        # 检查防守
        board.grid[row][col] = opponent
        if board.is_five_in_a_row(row, col):
            board.grid[row][col] = EMPTY
            return 900_000_000
        board.grid[row][col] = EMPTY
        
        # 棋型评估
        board.grid[row][col] = player
        own_score = self._evaluate_position(board, row, col, player)
        board.grid[row][col] = EMPTY
        
        board.grid[row][col] = opponent
        opp_score = self._evaluate_position(board, row, col, opponent)
        board.grid[row][col] = EMPTY
        
        # 历史启发加权
        history_bonus = self.history[player].get((row, col), 0) * 100
        
        return own_score + opp_score * 1.5 + history_bonus

    def _alpha_beta(self, board: GomokuBoard, depth: int, alpha: float, beta: float, player: int) -> int:
        """Alpha-Beta 搜索，集成置换表、杀手走法"""
        self.search_count += 1
        
        if self._timed_out():
            return 0

        # 终止条件：检查赢/负（而非硬编码在评估里）
        winner = board.winner()
        if winner == player:
            return 100_000 + (10 - depth)  # 越早赢越好
        if winner == self._opponent(player):
            return -100_000 - (10 - depth)  # 越晚输越好
        
        if depth == 0 or board.is_full():
            return self._evaluate(board, player)

        # 查置换表（持久缓存）
        zobrist_hash = board.zobrist.get_hash()
        alpha_orig = alpha
        
        if zobrist_hash in self.transposition:
            entry = self.transposition[zobrist_hash]
            if entry.depth >= depth:
                self.transposition_hits += 1
                if entry.flag == 'exact':
                    return entry.value
                elif entry.flag == 'lower':
                    alpha = max(alpha, entry.value)
                elif entry.flag == 'upper':
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value
        
        self.transposition_misses += 1

        moves = self._generate_moves(board, player)
        if not moves:
            return self._evaluate(board, player)

        # 杀手走法优先排序
        killer_list = self.killer_moves.get(depth, [])
        killer_set = set(killer_list)
        
        moves_sorted = []
        for move in moves:
            if move in killer_set:
                moves_sorted.insert(0, move)  # 杀手走法放在前面
            else:
                moves_sorted.append(move)

        value = -math.inf
        best_move = None
        
        for row, col in moves_sorted:
            board.play(row, col, player)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, self._opponent(player))
            board.undo()

            if score > value:
                value = score
                best_move = (row, col)
            
            alpha = max(alpha, value)
            if alpha >= beta:
                # 更新杀手走法
                if best_move and depth > 0:
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if best_move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, best_move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                    
                    # 更新历史启发
                    self.history[player][best_move] = self.history[player].get(best_move, 0) + (1 << depth)
                
                break
            
            if self._timed_out():
                break

        # 存入置换表
        flag = 'exact'
        if value <= alpha_orig:
            flag = 'upper'
        elif value >= beta:
            flag = 'lower'
        
        self.transposition[zobrist_hash] = TranspositionEntry(
            depth=depth,
            value=value,
            flag=flag,
            timestamp=self.timestamp
        )
        
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

    @staticmethod
    def _opponent(player: int) -> int:
        return BLACK if player == WHITE else WHITE


def play_gomoku_game():
    board = GomokuBoard(size=15)
    ai = GomokuAI(size=15, time_limit=2.0)

    print("=" * 50)
    print("    🎮 高级五子棋 AI (Zobrist优化版) 🎮")
    print("=" * 50)
    print("规则：你是黑方(X)，AI是白方(O)")
    print("落子输入格式：行 列 （例如：7 7 代表棋盘中心）")
    print("输入 'q' 可退出游戏")
    print("=" * 50)

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
                    print(f"你落子：{x} {y}")
                    break
                else:
                    print("❌ 坐标无效，请选择空白位置！")
            except ValueError:
                print("❌ 输入格式错误，请输入两个数字，空格分隔！")

        if board.winner() or board.is_full():
            continue

        # AI 落子
        print("\n🤖 AI 思考中...")
        ai_row, ai_col = ai.best_move(board, WHITE)
        board.play(ai_row, ai_col, WHITE)
        print(f"AI 落子：{ai_row} {ai_col}")
        print()


def run_example() -> None:
    """运行示例"""
    print("\n" + "=" * 50)
    print("     📊 AI 走法推荐示例 (Zobrist优化版) 📊")
    print("=" * 50)
    board = GomokuBoard(size=15)
    ai = GomokuAI(size=15, time_limit=2.0)
    
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
