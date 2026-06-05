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

# 控制台交互式下棋（原生无依赖，直接运行就能手动下棋）
if __name__ == "__main__":
    print("✅ KataGo五子棋AI启动 | 输入格式：x  y（空格分隔，例：7 7）")
    while True:
        try:
            user_input = input("请输入你的落子坐标：")
            x, y = map(int, user_input.strip().split())
            human_put(x, y)
            ax, ay = ai_get_move()
            print(f"🤖 AI落子：{ax} , {ay}\n")
        except:
            print("❌ 输入格式错误！请输入两个数字，中间空格隔开\n")
