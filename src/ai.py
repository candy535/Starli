import random

class IllegalMoveError(ValueError):
    pass
Pos = int
Loc = int
Player = int

class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    WALL = 3
    ZOBRIST_STONE = [[],[],[],[]]
    ZOBRIST_PLA = []
    ZOBRIST_RAND = random.Random()
    ZOBRIST_RAND.seed(123987456)
    PASS_LOC = 0
    for i in range((50+1)*(50+2)+1):
        ZOBRIST_STONE[BLACK].append(ZOBRIST_RAND.getrandbits(64))
        ZOBRIST_STONE[WHITE].append(ZOBRIST_RAND.getrandbits(64))
    for i in range(4):
        ZOBRIST_PLA.append(ZOBRIST_RAND.getrandbits(64))

    def __init__(self,size):
        if isinstance(size,int):
            self.x_size = size
            self.y_size = size
        else:
            self.x_size, self.y_size = size
        self.arrsize = (self.x_size+1)*(self.y_size+2)+1
        self.dy = self.x_size+1
        self.adj = [-self.dy,-1,1,self.dy]
        self.diag = [-self.dy-1,-self.dy+1,self.dy-1,self.dy+1]
        self.pla = Board.BLACK
        self.board = [Board.EMPTY]*self.arrsize
        self.group_head = [0]*self.arrsize
        self.group_stone_count = [0]*self.arrsize
        self.group_liberty_count = [0]*self.arrsize
        self.group_next = [0]*self.arrsize
        self.group_prev = [0]*self.arrsize
        self.zobrist = 0
        self.simple_ko_point = None
        self.num_captures_made = {Board.BLACK:0, Board.WHITE:0}
        self.num_non_pass_moves_made = {Board.BLACK:0, Board.WHITE:0}
        for i in range(-1,self.x_size+1):
            self.board[self.loc(i,-1)] = Board.WALL
            self.board[self.loc(i,self.y_size)] = Board.WALL
        for i in range(-1,self.y_size+1):
            self.board[self.loc(-1,i)] = Board.WALL
            self.board[self.loc(self.x_size,i)] = Board.WALL
        self.group_head[0] = -1
        self.group_next[0] = -1
        self.group_prev[0] = -1

    def copy(self):
        newb = Board((self.x_size,self.y_size))
        newb.pla = self.pla
        newb.board = self.board.copy()
        newb.group_head = self.group_head.copy()
        newb.group_stone_count = self.group_stone_count.copy()
        newb.group_liberty_count = self.group_liberty_count.copy()
        newb.group_next = self.group_next.copy()
        newb.group_prev = self.group_prev.copy()
        newb.zobrist = self.zobrist
        newb.simple_ko_point = self.simple_ko_point
        newb.num_captures_made = self.num_captures_made.copy()
        newb.num_non_pass_moves_made = self.num_non_pass_moves_made.copy()
        return newb

    @staticmethod
    def get_opp(pla):
        return 3-pla
    def loc(self,x,y):
        return (x+1)+self.dy*(y+1)
    def loc_x(self,loc):
        return (loc%self.dy)-1
    def loc_y(self,loc):
        return (loc//self.dy)-1
    def is_on_board(self,loc):
        return 0<=loc<self.arrsize and self.board[loc]!=Board.WALL
    def would_be_legal(self,pla,loc):
        if pla not in (Board.BLACK,Board.WHITE):return False
        if loc==Board.PASS_LOC:return True
        if not self.is_on_board(loc):return False
        if self.board[loc]!=Board.EMPTY:return False
        return True
    def play(self,pla,loc):
        if not self.would_be_legal(pla,loc):
            raise IllegalMoveError()
        if loc!=Board.PASS_LOC:
            self.board[loc]=pla
        self.pla = Board.get_opp(pla)

class GameState:
    RULES_TT = {}
    def __init__(self,bsize,rule):
        self.board = Board(bsize)
    def play(self,pla,loc):
        self.board.play(pla,loc)
    def search_move_no_nn(self):
        b=self.board
        sz=b.x_size
        lst=[]
        for x in range(sz):
            for y in range(sz):
                l=b.loc(x,y)
                if b.would_be_legal(b.pla,l):
                    lst.append(l)
        if not lst:return Board.PASS_LOC
        return random.choice(lst)

#AI对外接口
BOARD_SIZE=15
gs=GameState(BOARD_SIZE,GameState.RULES_TT)
def human_put(x,y,is_black=True):
    p=Board.BLACK if is_black else Board.WHITE
    pos=gs.board.loc(x,y)
    gs.play(p,pos)
def ai_get_move():
    pos=gs.search_move_no_nn()
    x=gs.board.loc_x(pos)
    y=gs.board.loc_y(pos)
    gs.play(gs.board.pla,pos)
    return x,y
