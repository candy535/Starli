from typing import Dict, Any, List, Tuple, Union, Optional
import math
import numpy as np

# 导入改造后的五子棋棋盘
from game.board import Board

# 临时占位：后续对接 KataGo 五子棋特征/模型
class Features:
    pass
class SGFMetadata:
    pass

if __name__ != "__main__":
    from train.model_pytorch import Model

class GameState:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.board = Board(size=board_size)
        self.moves = []       # 历史落子
        self.boards = [self.board.copy()]
        self.redo_stack = []  # 重做栈

    def play(self, pla, loc):
        """执行一步落子"""
        self.board.play(pla, loc)
        self.moves.append((pla, loc))
        self.boards.append(self.board.copy())
        self.redo_stack.clear()

    # 悔棋 / 重做
    def can_undo(self) -> bool:
        return len(self.moves) > 0

    def undo(self):
        assert self.can_undo()
        move = self.moves.pop()
        self.boards.pop()
        self.board = self.boards[-1].copy()
        self.redo_stack.append(move)

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def redo(self):
        assert self.can_redo()
        move = self.redo_stack.pop()
        self.moves.append(move)
        self.boards.append(self.board.copy())
        self.board = self.boards[-1].copy()

    # ========== 预留：对接五子棋 KataGo 模型特征（后续核心） ==========
    def get_input_features(self, features: Features):
        """预留：将棋盘转为神经网络输入特征图"""
        pos_len = self.board_size
        bin_input_data = np.zeros([1, pos_len, pos_len, 3], dtype=np.float32)
        # 简单特征：黑棋通道、白棋通道、空点通道
        for y in range(pos_len):
            for x in range(pos_len):
                loc = self.board.loc(x, y)
                val = self.board.board[loc]
                if val == Board.BLACK:
                    bin_input_data[0, y, x, 0] = 1.0
                elif val == Board.WHITE:
                    bin_input_data[0, y, x, 1] = 1.0
                else:
                    bin_input_data[0, y, x, 2] = 1.0
        global_input_data = np.zeros([1, 2], dtype=np.float32)
        global_input_data[0, 0] = 1.0 if self.board.pla == Board.BLACK else 0.0
        return bin_input_data, global_input_data

    # ========== 预留：KataGo 模型推理 + 生成AI落子 ==========
    def get_model_outputs(self, model: "Model"):
        """调用KataGo模型，返回AI推荐落子（后续对接官方权重）"""
        import torch
        features = Features()
        bin_feat, global_feat = self.get_input_features(features)

        with torch.no_grad():
            model.eval()
            in1 = torch.tensor(bin_feat, dtype=torch.float32, device=model.device)
            in2 = torch.tensor(global_feat, dtype=torch.float32, device=model.device)
            model_out = model(in1, in2)

        # 临时逻辑：遍历所有合法点，后续替换为模型Policy
        legal_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                loc = self.board.loc(x, y)
                if self.board.would_be_legal(self.board.pla, loc):
                    legal_moves.append(loc)

        # 临时随机走子（仅测试规则，棋力极低）
        import random
        genmove = random.choice(legal_moves) if legal_moves else Board.PASS_LOC

        return {
            "genmove_result": genmove,
            "legal_moves": legal_moves
        }
