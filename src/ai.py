from src.board import Board
from src.gamestate import GameState

# 15×15五子棋棋盘
BOARD_SIZE = 15
gs = GameState(BOARD_SIZE, GameState.RULES_TT)

# 人类落子：x横向，y纵向
def human_put(x,y,is_black=True):
    pla = Board.BLACK if is_black else Board.WHITE
    loc = gs.board.loc(x,y)
    gs.play(pla,loc)

# KataGo纯MCTS AI落子（不用神经网络、不用权重）
def ai_get_move(is_black=True):
    pos = gs.search_move_no_nn()
    pla = Board.BLACK if is_black else Board.WHITE
    gs.play(pla,pos)
    x = gs.board.loc_x(pos)
    y = gs.board.loc_y(pos)
    return x,y

# 本地测试代码
if __name__ == "__main__":
    human_put(7,7)    # 人下天元(7,7)
    ax,ay = ai_get_move()
    print("AI落子坐标：",ax,ay)
