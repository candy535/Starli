import numpy as np
import time
import random

# ---------- Zobrist 哈希表（用于置换表，防重复计算） ----------
class Zobrist:
    def __init__(self, size=15):
        self.size = size
        # 每个位置：空/黑/白 三种状态随机哈希值
        self.positions = np.random.randint(1, 2**63, (size, size, 3), dtype=np.int64)
        self.current = np.random.randint(1, 2**63, dtype=np.int64)

    def hash_board(self, board, current_player):
        h = 0
        for r in range(self.size):
            for c in range(self.size):
                val = board[r, c]
                if val != 0:
                    h ^= self.positions[r, c, val]
        if current_player == 2:
            h ^= self.current
        return h

# ---------- 棋盘类：落子、悔棋、判赢、威胁检测 ----------
class GomokuBoard:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int32)  # 0空 1黑 2白
        self.current = 1          # 黑先行
        self.zob = Zobrist(size)
        self.history = []         # 落子历史，用于悔棋
        self.tt = {}              # 置换表 {哈希值: (分值, 搜索深度)}

    def copy(self):
        new_board = GomokuBoard(self.size)
        new_board.board = self.board.copy()
        new_board.current = self.current
        new_board.history = list(self.history)
        new_board.tt = {}
        return new_board

    def move(self, x, y, player=None):
        if player is None:
            player = self.current
        if self.board[x, y] != 0:
            return False
        self.board[x, y] = player
        self.history.append((x, y, player))
        self.current = 3 - self.current
        return True

    def undo(self):
        if not self.history:
            return
        x, y, player = self.history.pop()
        self.board[x, y] = 0
        self.current = player

    def is_win(self, player):
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] != player:
                    continue
                for dx, dy in dirs:
                    cnt = 1
                    nx, ny = x + dx, y + dy
                    while 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                        cnt += 1
                        nx += dx
                        ny += dy
                    if cnt >= 5:
                        return True
        return False

    def get_legal_moves(self):
        return
