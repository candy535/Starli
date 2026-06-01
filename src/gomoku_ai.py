"""
A stronger Gomoku AI engine with board heuristics and alpha-beta search.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

PATTERN_SCORES = {
    (5, 0): 1_000_000_000,
    (4, 0): 100_000_000,
    (4, 1): 10_000_000,
    (3, 0): 1_000_000,
    (3, 1): 100_000,
    (2, 0): 1_000,
    (2, 1): 200,
    (1, 0): 50,
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
        new_board.count = self.count
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
        header = '  ' + ' '.join(f'{i:2}' for i in range(self.size))
        print(header)
        for i, row in enumerate(self.grid):
            print(f'{i:2} ' + ' '.join('.' if cell == EMPTY else 'X' if cell == BLACK else 'O' for cell in row))

class GomokuAI:
    def __init__(self, size: int = 15, time_limit: float = 2.0) -> None:
        self.size = size
        self.time_limit = time_limit
        self.start_time = 0.0
        self.transposition: dict[Tuple[Tuple[int, ...], int], int] = {}

    def best_move(self, board: GomokuBoard, player: int) -> Tuple[int, int]:
        self.start_time = time.time()
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
        return best_move

    def _timed_out(self) -> bool:
        return time.time() - self.start_time > self.time_limit

    def _select_depth(self, board: GomokuBoard) -> int:
        if board.count < 10:
            return 4
        if board.count < 40:
            return 5
        return 6

    def _generate_moves(self, board: GomokuBoard, player: int) -> List[Tuple[int, int]]:
        candidates = board.get_neighbors(distance=2)
        scored_moves = []
        for row, col in candidates:
            score = self._move_heuristic(board, row, col, player)
            scored_moves.append(((row, col), score))
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        return [move for move, _ in scored_moves[:min(30, len(scored_moves))]]

    def _move_heuristic(self, board: GomokuBoard, row: int, col: int, player: int) -> int:
        board.grid[row][col] = player
        score = self._evaluate_position(board, row, col, player)
        board.grid[row][col] = EMPTY
        return score

    def _alpha_beta(self, board: GomokuBoard, depth: int, alpha: float, beta: float, player: int):
        if self._timed_out():
            return 0.0
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
            value = max(value, score)
            alpha = max(alpha, score)
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
            blocks = 0
            idx = start
            while idx < n and line[idx] == player:
                count += 1
                idx += 1
            left_empty = start - 1 >= 0 and line[start - 1] == EMPTY
            right_empty = idx < n and line[idx] == EMPTY
            if start - 1 < 0 or start - 1 >= n or line[start - 1] != EMPTY:
                blocks += 1
            if idx >= n or line[idx] != EMPTY:
                blocks += 1
            pattern_score = self._pattern_score(count, blocks, left_empty, right_empty)
            best = max(best, pattern_score)
        return best

    def _pattern_score(self, count: int, blocks: int, left_open: bool, right_open: bool) -> int:
        if count >= 5:
            return PATTERN_SCORES[(5, 0)]
        open_ends = int(left_open) + int(right_open)
        if count == 4:
            if open_ends == 2:
                return PATTERN_SCORES[(4, 0)]
            if open_ends == 1:
                return PATTERN_SCORES[(4, 1)]
        elif count == 3:
            if open_ends == 2:
                return PATTERN_SCORES[(3, 0)]
            if open_ends == 1:
                return PATTERN_SCORES[(3, 1)]
        elif count == 2:
            if open_ends == 2:
                return PATTERN_SCORES[(2, 0)]
            if open_ends == 1:
                return PATTERN_SCORES[(2, 1)]
        elif count == 1:
            if open_ends == 2:
                return PATTERN_SCORES[(1, 0)]
        return 0

    def _hash_board(self, board: GomokuBoard, player: int) -> Tuple[Tuple[int, ...], int]:
        flat = tuple(cell for row in board.grid for cell in row)
        return flat, player

    @staticmethod
    def _opponent(player: int) -> int:
        return BLACK if player == WHITE else WHITE

def run_example() -> None:
    board = GomokuBoard(size=15)
    ai = GomokuAI(size=15, time_limit=1.5)
    starting_moves = [
        (7, 7, BLACK),
        (7, 8, WHITE),
        (8, 7, BLACK),
        (8, 8, WHITE),
    ]
    for row, col, player in starting_moves:
        board.play(row, col, player)

    board.print_board()
    move = ai.best_move(board, BLACK)
    print(f"AI recommends move: {move}")

if __name__ == "__main__":
    run_example()
