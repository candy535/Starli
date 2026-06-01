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
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]

    # 威胁检测：冲四、活三、眠三
    def check_lines(self, player):
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        threats = []
        opp = 3 - player
        size = self.size
        for x in range(size):
            for y in range(size):
                if self.board[x, y] != 0:
                    continue
                for dx, dy in dirs:
                    line = []
                    # 取当前点前后共9格窗口
                    for i in range(-4, 5):
                        nx = x + i * dx
                        ny = y + i * dy
                        if 0 <= nx < size and 0 <= ny < size:
                            line.append(self.board[nx, ny])
                        else:
                            line.append(-1)
                    # 模拟在此处落子
                    temp_line = line.copy()
                    temp_line[4] = player
                    max_own = 0
                    # 滑动5格窗口检测连子
                    for s in range(len(temp_line) - 4):
                        seg = temp_line[s:s+5]
                        if -1 in seg:
                            continue
                        own = seg.count(player)
                        enemy = seg.count(opp)
                        if enemy > 0:
                            continue
                        if own > max_own:
                            max_own = own
                    # 判定威胁类型
                    if max_own == 4:
                        threats.append(((x, y), 1, (dx, dy)))   # 冲四
                    elif max_own == 3:
                        # 简单判断活三/眠三
                        left = (x - 4*dx, y - 4*dy)
                        right = (x + dx, y + dy)
                        left_open = 0<=left[0]<size and 0<=left[1]<size and self.board[left] == 0
                        right_open = 0<=right[0]<size and 0<=right[1]<size and self.board[right] == 0
                        if left_open and right_open:
                            threats.append(((x, y), 2, (dx, dy))) # 活三
                        else:
                            threats.append(((x, y), 0.5, (dx, dy)))# 眠三
        return threats

    # 评估当前盘面分数
    def evaluate(self):
        if self.is_win(1):
            return 1000000
        if self.is_win(2):
            return -1000000

        black_th = self.check_lines(1)
        white_th = self.check_lines(2)
        b_score = 0
        w_score = 0

        # 威胁打分
        for _, t, _ in black_th:
            if t == 1:
                b_score += 10000
            elif t == 2:
                b_score += 500
        for _, t, _ in white_th:
            if t == 1:
                w_score += 10000
            elif t == 2:
                w_score += 500

        # 位置分：中心位置加分
        center = (self.size - 1) / 2
        for r in range(self.size):
            for c in range(self.size):
                dist = (r - center)**2 + (c - center)**2
                if self.board[r,c] == 1:
                    b_score += 20 - dist * 0.5
                elif self.board[r,c] == 2:
                    w_score += 20 - dist * 0.5

        return b_score - w_score

    # 走法排序：优先防守对方杀招，再进攻，最后普通点位
    def ordered_moves(self):
        moves = self.get_legal_moves()
        curr = self.current
        opp = 3 - curr
        opp_threats = self.check_lines(opp)

        def move_score(m):
            x, y = m
            # 堵冲四 > 堵活三 > 普通点位
            for (px, py), t, _ in opp_threats:
                if (px, py) == (x, y):
                    if t == 1:
                        return 10000
                    elif t == 2:
                        return 5000
            # 靠近中心加分
            center = (self.size-1)/2
            return 500 - abs(x-center) - abs(y-center)

        # 分数从高到低排序
        moves.sort(key=move_score, reverse=True)
        return moves

# ---------- Alpha-Beta 剪枝 AI 核心 ----------
class AlphaBetaAI:
    def __init__(self, board):
        self.board = board

    def alpha_beta(self, board, depth, alpha, beta, maximizing):
        # 1. 置换表查询，避免重复计算同一盘面
        h = board.zob.hash_board(board.board, board.current)
        if h in board.tt:
            val, d = board.tt[h]
            if d >= depth:
                return val

        # 递归终止：到底 / 分出胜负
        if board.is_win(1):
            return 1000000
        if board.is_win(2):
            return -1000000
        if depth <= 0:
            return board.evaluate()

        moves = board.ordered_moves()

        # 极大层（黑方，追求最高分）
        if maximizing:
            max_val = -float('inf')
            for mv in moves:
                board.move(*mv)
                res = self.alpha_beta(board, depth-1, alpha, beta, False)
                board.undo()
                max_val = max(max_val, res)
                alpha = max(alpha, max_val)
                # ✂️ Alpha-Beta 剪枝核心：beta <= alpha 直接截断
                if beta <= alpha:
                    break
            board.tt[h] = (max_val, depth)
            return max_val
        # 极小层（白方，追求最低分）
        else:
            min_val = float('inf')
            for mv in moves:
                board.move(*mv)
                res = self.alpha_beta(board, depth-1, alpha, beta, True)
                board.undo()
                min_val = min(min_val, res)
                beta = min(beta, min_val)
                # ✂️ 剪枝
                if beta <= alpha:
                    break
            board.tt[h] = (min_val, depth)
            return min_val

    # 迭代加深搜索 + 限时控制，选出最优落子
    def best_move(self, time_limit=1.0):
        board_cpy = self.board.copy()
        best_mv = None
        start_t = time.time()
        # 逐层加深搜索深度（1~9层）
        for d in range(1, 10):
            if time.time() - start_t > time_limit:
                break
            current_best = None
            if board_cpy.current == 1:
                best_v = -float('inf')
                for mv in board_cpy.ordered_moves():
                    board_cpy.move(*mv)
                    v = self.alpha_beta(board_cpy, d-1, -float('inf'), float('inf'), False)
                    board_cpy.undo()
                    if v > best_v:
                        best_v = v
                        current_best = mv
            else:
                best_v = float('inf')
                for mv in board_cpy.ordered_moves():
                    board_cpy.move(*mv)
                    v = self.alpha_beta(board_cpy, d-1, -float('inf'), float('inf'), True)
                    board_cpy.undo()
                    if v < best_v:
                        best_v = v
                        current_best = mv
            if current_best:
                best_mv = current_best
        return best_mv

# ---------- 运行对局：AI(黑) VS 随机玩家(白) ----------
if __name__ == "__main__":
    game = GomokuBoard()
    ai = AlphaBetaAI(game)

    while True:
        if game.is_win(1):
            print("🏆 黑方(AI) 获胜！")
            break
        if game.is_win(2):
            print("🏆 白方(玩家) 获胜！")
            break
        if not game.get_legal_moves():
            print("🤝 平局！")
            break

        if game.current == 1:
            # AI 落子
            mv = ai.best_move(0.8)
            if mv is None:
                print("AI无合法走法，平局")
                break
            game.move(*mv)
            print(f"AI 落子: {mv}")
        else:
            # 模拟人类：随机落子
            legal = game.get_legal_moves()
            mv = random.choice(legal)
            game.move(*mv)
            print(f"玩家 落子: {mv}")
