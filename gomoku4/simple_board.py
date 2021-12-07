"""
simple_board.py

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from board_util import GoBoardUtil, BLACK, WHITE, EMPTY, BORDER, \
                       PASS, is_black_white, coord_to_point, where1d, \
                       MAXSIZE, NULLPOINT
import alphabeta

class SimpleGoBoard(object):

    def get_color(self, point):
        return self.board[point]

    def pt(self, row, col):
        return coord_to_point(row, col, self.size)

    def is_legal(self, point, color):
        """
        Check whether it is legal for color to play on point
        """
        assert is_black_white(color)
        # Special cases
        if point == PASS:
            return True
        elif self.board[point] != EMPTY:
            return False
        if point == self.ko_recapture:
            return False
            
        # General case: detect captures, suicide
        opp_color = GoBoardUtil.opponent(color)
        self.board[point] = color
        legal = True
        has_capture = self._detect_captures(point, opp_color)
        if not has_capture and not self._stone_has_liberty(point):
            block = self._block_of(point)
            if not self._has_liberty(block): # suicide
                legal = False
        self.board[point] = EMPTY
        return legal

    def _detect_captures(self, point, opp_color):
        """
        Did move on point capture something?
        """
        for nb in self.neighbors_of_color(point, opp_color):
            if self._detect_capture(nb):
                return True
        return False

    def get_empty_points(self):
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def __init__(self, size):
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)

    def reset(self, size):
        """
        Creates a start state, an empty board with the given size
        The board is stored as a one-dimensional array
        See GoBoardUtil.coord_to_point for explanations of the array encoding
        """
        self.size = size
        self.NS = size + 1
        self.WE = 1
        self.ko_recapture = None
        self.current_player = BLACK
        self.maxpoint = size * size + 3 * (size + 1)
        self.board = np.full(self.maxpoint, BORDER, dtype = np.int32)
        self.liberty_of = np.full(self.maxpoint, NULLPOINT, dtype = np.int32)
        self._initialize_empty_points(self.board)
        self._initialize_neighbors()

    def copy(self):
        b = SimpleGoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def row_start(self, row):
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1
        
    def _initialize_empty_points(self, board):
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start = self.row_start(row)
            board[start : start + self.size] = EMPTY

    def _on_board_neighbors(self, point):
        nbs = []
        for nb in self._neighbors(point):
            if self.board[nb] != BORDER:
                nbs.append(nb)
        return nbs
            
    def _initialize_neighbors(self):
        """
        precompute neighbor array.
        For each point on the board, store its list of on-the-board neighbors
        """
        self.neighbors = []
        for point in range(self.maxpoint):
            if self.board[point] == BORDER:
                self.neighbors.append([])
            else:
                self.neighbors.append(self._on_board_neighbors(point))
        
    def is_eye(self, point, color):
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = GoBoardUtil.opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge # 0 at edge, 1 in center
    
    def _is_surrounded(self, point, color):
        """
        check whether empty point is surrounded by stones of color.
        """
        for nb in self.neighbors[point]:
            nb_color = self.board[nb]
            if nb_color != color:
                return False
        return True

    def _stone_has_liberty(self, stone):
        lib = self.find_neighbor_of_color(stone, EMPTY)
        return lib != None

    def _get_liberty(self, block):
        """
        Find any liberty of the given block.
        Returns None in case there is no liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            lib = self.find_neighbor_of_color(stone, EMPTY)
            if lib != None:
                return lib
        return None

    def _has_liberty(self, block):
        """
        Check if the given block has any liberty.
        Also updates the liberty_of array.
        block is a numpy boolean array
        """
        lib = self._get_liberty(block)
        if lib != None:
            assert self.get_color(lib) == EMPTY
            for stone in where1d(block):
                self.liberty_of[stone] = lib
            return True
        return False

    def _block_of(self, stone):
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        marker = np.full(self.maxpoint, False, dtype = bool)
        pointstack = [stone]
        color = self.get_color(stone)
        assert is_black_white(color)
        marker[stone] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _fast_liberty_check(self, nb_point):
        lib = self.liberty_of[nb_point]
        if lib != NULLPOINT and self.get_color(lib) == EMPTY:
            return True # quick exit, block has a liberty  
        if self._stone_has_liberty(nb_point):
            return True # quick exit, no need to look at whole block
        return False
        
    def _detect_capture(self, nb_point):
        """
        Check whether opponent block on nb_point is captured.
        Returns boolean.
        """
        if self._fast_liberty_check(nb_point):
            return False
        opp_block = self._block_of(nb_point)
        return not self._has_liberty(opp_block)
    
    def _detect_and_process_capture(self, nb_point):
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
            and returns None otherwise.
        This result is used in play_move to check for possible ko
        """
        if self._fast_liberty_check(nb_point):
            return None
        opp_block = self._block_of(nb_point)
        if self._has_liberty(opp_block):
            return None
        captures = list(where1d(opp_block))
        self.board[captures] = EMPTY
        self.liberty_of[captures] = NULLPOINT
        single_capture = None 
        if len(captures) == 1:
            single_capture = nb_point
        return single_capture
# The above code is used for Go.
    def play_move(self, point, color):
        """
        Play a move of color on point
        Returns boolean: whether move was legal
        """
        assert is_black_white(color)
        # Special cases
        if point == PASS:
            self.ko_recapture = None
            self.current_player = GoBoardUtil.opponent(color)
            return True
        elif self.board[point] != EMPTY:
            return False
        if point == self.ko_recapture:
            return False
            
        # General case: deal with captures, suicide, and next ko point
        opp_color = GoBoardUtil.opponent(color)
        in_enemy_eye = self._is_surrounded(point, opp_color)
        self.board[point] = color
        single_captures = []
        neighbors = self.neighbors[point]
        for nb in neighbors:
            if self.board[nb] == opp_color:
                single_capture = self._detect_and_process_capture(nb)
                if single_capture != None:
                    single_captures.append(single_capture)
        if not self._stone_has_liberty(point):
            # check suicide of whole block
            block = self._block_of(point)
            if not self._has_liberty(block): # undo suicide move
                self.board[point] = EMPTY
                return False
        self.ko_recapture = None
        if in_enemy_eye and len(single_captures) == 1:
            self.ko_recapture = single_captures[0]
        self.current_player = GoBoardUtil.opponent(color)
        return True

    def neighbors_of_color(self, point, color):
        """ List of neighbors of point of given color """
        nbc = []
        for nb in self.neighbors[point]:
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc
        
    def find_neighbor_of_color(self, point, color):
        """ Return one neighbor of point of given color, or None """
        for nb in self.neighbors[point]:
            if self.get_color(nb) == color:
                return nb
        return None
        
    def _neighbors(self, point):
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point):
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1, 
                point - self.NS + 1, 
                point + self.NS - 1, 
                point + self.NS + 1]
    
    def _point_to_coord(self, point):
        """
        Transform point index to row, col.
        
        Arguments
        ---------
        point
        
        Returns
        -------
        x , y : int
        coordination of the board  1<= x <=size, 1<= y <=size .
        """
        if point is None:
            return 'pass'
        row, col = divmod(point, self.NS)
        return row, col

    def is_legal_gomoku(self, point, color):
        """
            Check whether it is legal for color to play on point, for the game of gomoku
            """
        return self.board[point] == EMPTY
    
    def play_move_gomoku(self, point, color):
        """
            Play a move of color on point, for the game of gomoku
            Returns boolean: whether move was legal
            """
        assert is_black_white(color)
        assert point != PASS
        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        self.current_player = GoBoardUtil.opponent(color)
        return True
        
    def _point_direction_check_connect_gomoko(self, point, shift):
        """
        Check if the point has connect5 condition in a direction
        for the game of Gomoko.
        """
        color = self.board[point]
        count = 1
        d = shift
        p = point
        while True:
            p = p + d
            if self.board[p] == color:
                count = count + 1
                if count == 5:
                    break
            else:
                break
        d = -d
        p = point
        while True:
            p = p + d
            if self.board[p] == color:
                count = count + 1
                if count == 5:
                    break
            else:
                break
        assert count <= 5
        return count == 5
    
    def point_check_game_end_gomoku(self, point):
        """
            Check if the point causes the game end for the game of Gomoko.
            """
        # check horizontal
        if self._point_direction_check_connect_gomoko(point, 1):
            return True
        
        # check vertical
        if self._point_direction_check_connect_gomoko(point, self.NS):
            return True
        
        # check y=x
        if self._point_direction_check_connect_gomoko(point, self.NS + 1):
            return True
        
        # check y=-x
        if self._point_direction_check_connect_gomoko(point, self.NS - 1):
            return True
        
        return False
    
    def check_game_end_gomoku(self):
        """
            Check if the game ends for the game of Gomoku.
            """
        white_points = where1d(self.board == WHITE)
        black_points = where1d(self.board == BLACK)
        
        for point in white_points:
            if self.point_check_game_end_gomoku(point):
                return True, WHITE
    
        for point in black_points:
            if self.point_check_game_end_gomoku(point):
                return True, BLACK

        return False, None

    def solve(self):
        result, move, drawMove = alphabeta.solve(self)
        if move=="First":
            if result==0:
                return 'draw',drawMove
            else:
                winner='w' if self.current_player!=WHITE else 'b'
                return winner,'NoMove'
        elif move=="NoMove":
            if result:
                return 'draw', drawMove
            else:
                winner='w' if self.current_player!=WHITE else 'b'
                return winner, move
        else:
            winner='w' if self.current_player==WHITE else 'b'
            return winner, move

    def check_pattern(self,point,have,direction_x,direction_y,moveSet,patternList,color,flag):
        for i in range(0,4):
            if have in patternList[i]:
                for dis in patternList[i][have]:
                    moveSet[i].add(point-direction_x*(dis+1)-direction_y*self.NS*(dis+1))
                #flag[0]=True
                break
        if (not (0<= point<len(self.board))) or len(have)==9:
            return
#if self.get_color(point)==BORDER or len(have)==7:
#            return
        piece=self.get_color(point)
        if piece==EMPTY:
            piece='.'
        elif piece==color:
            piece='x'
        elif piece == BORDER:
            piece='B'
        else:
            piece='o'
        have+=piece
        #print(GoBoardUtil.format_point(self._point_to_coord(point)),have,self.board[point])
        self.check_pattern(point+direction_x+direction_y*self.NS,have,direction_x,direction_y,moveSet,patternList,color,flag)

    def get_pattern_moves(self):
        """
        1. direct winning point xxxx. x.xxx xx.xx
        2. urgent blocking point xoooo.
        3. wining in 2 step point
        """
        moveSet=[set(),set(),set(),set()]
        color=self.current_player

        patternList=[{'xxxx.':{0},'xxx.x':{1},'xx.xx':{2},'x.xxx':{3},'.xxxx':{4}}, #win
                     {'oooo.':{0},'ooo.o':{1},'oo.oo':{2},'o.ooo':{3},'.oooo':{4}}, #block win
                     {'.xxx..':{1},'..xxx.':{4},'.xx.x.':{2},'.x.xx.':{3}}, #make-four
                     {'.ooo..':{1,5},'..ooo.':{0,4},'.oo.o.':{0,2,5},'.o.oo.':{0,3,5}, 'B.ooo..':{0}, '..ooo.B':{6},
                     'x.ooo..':{0}, '..ooo.x':{6} #block-open-four
                     }]

        direction_x=[1,0,1,-1]
        direction_y=[0,1,1,1]
        flag=[False]

        for point in range(0, len(self.board)):
            if flag[0]:
                break
            for direction in range(0,4):
                    self.check_pattern(point,'',direction_x[direction],direction_y[direction],moveSet,patternList,color,flag)
        
        i=0
        while i<4 and not bool(moveSet[i]): i+=1
        if i==4:
            return None
        else:
            return i, list(moveSet[i])
            
    def list_solve_point(self):
        """
        1. direct winning point xxxx. x.xxx xx.xx
        2. urgent blocking point xoooo.
        3. wining in 2 step point
        """
        moveSet=[set(),set(),set(),set()]
        color=self.current_player

        patternList=[{'xxxx.':{0},'xxx.x':{1},'xx.xx':{2},'x.xxx':{3},'.xxxx':{4}},{'oooo.':{0},'ooo.o':{1},'oo.oo':{2},'o.ooo':{3},'.oooo':{4}},{'.xxx..':{1},'..xxx.':{4},'.xx.x.':{2},'.x.xx.':{3}},{'.ooo..':{1,5},'..ooo.':{0,4},'.oo.o.':{2},'.o.oo.':{3}}]

        direction_x=[1,0,1,-1]
        direction_y=[0,1,1,1]
        flag=[False]

        for point in where1d(self.board!=BORDER):
            if flag[0]:
                break
            for direction in range(0,4):
                    self.check_pattern(point,'',direction_x[direction],direction_y[direction],moveSet,patternList,color,flag)
        
        i=0
        while i<4 and not bool(moveSet[i]):
            i+=1
        if i==4:
            return None
        else:
            return list(moveSet[i])
#
#
#
#
#
#
#
# From here, it is the beginning of alphabeta algorithm. 
#
#
#
#
#
#
    def check_heuristic_dict(self, point, counter, dictionary):
        if point != -1:
            if point not in dictionary.keys():
                dictionary[point] = counter
            elif point in dictionary.keys():
                if counter > dictionary[point]:
                    dictionary[point] = counter

    def mapping_player_heuristic(self, color):

        player_stone_list = [point for point in where1d(self.board == color)]
        if len(player_stone_list) == 0:
            player_stone_list = [
                point for point in where1d(self.board == EMPTY)
            ]

        potential_move_dict = {}

        for point in player_stone_list:
            potential_moves = []

            left_neighbor, right_neighbor, counter = self.analyze_hor(
                point, color)

            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])
            left_neighbor, right_neighbor, counter = self.analyze_ver(
                point, color)
            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])
            left_neighbor, right_neighbor, counter = self.analyze_left_diag(
                point, color)
            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])
            left_neighbor, right_neighbor, counter = self.analyze_right_diag(
                point, color)
            potential_moves.append([left_neighbor, counter])
            potential_moves.append([right_neighbor, counter])

            for value_pair in potential_moves:
                if value_pair[0] not in potential_move_dict.keys(
                ) and value_pair[0] != -1:
                    potential_move_dict[value_pair[0]] = value_pair[1]
                elif value_pair[0] in potential_move_dict.keys():
                    if value_pair[1] > potential_move_dict[value_pair[0]]:
                        potential_move_dict[value_pair[0]] = value_pair[1]

        
        return potential_move_dict

    def mapping_all_heuristic(self, color):
        ret = self.get_pattern_moves()
        movetype_id, moves=ret
    def alphabeta(self, color, alpha, beta, current_depth):
        player_dict, opponent_dict = self.mapping_all_heuristic(color)
        color_point = []
        player_count = len(list(player_dict.keys()))
        opponent_count = len(list(opponent_dict.keys()))

        while player_count > 0 and opponent_count > 0:
            player_key, player_value = max(player_dict.items(),
                                           key=lambda p: p[1])
            opponent_key, opponent_value = max(opponent_dict.items(),
                                               key=lambda p: p[1])
            if player_value >= opponent_value:

                if player_key not in color_point:

                    color_point.append(player_key)
                    player_dict.pop(player_key, None)
                    player_count -= 1
                else:
                    player_dict.pop(player_key, None)
                    player_count -= 1
            elif player_value < opponent_key:

                if opponent_key not in color_point:

                    color_point.append(opponent_key)
                    opponent_dict.pop(opponent_key, None)
                    opponent_count -= 1
                else:
                    opponent_dict.pop(opponent_key, None)
                    opponent_count -= 1
            else:
                break

        
        opponent = WHITE + BLACK - color
        
        EMPTY_list = []
        self.winning_move = None
        _, winner_color = self.check_game_end_gomoku()
        if winner_color == color:
            self.winner = color
            return 1
        elif winner_color == opponent:
            self.winner = opponent
            return -1
        if current_depth == self.depth or len(
                where1d(self.board == EMPTY).reshape(1, -1)[0]) == 0:
            return 0

        for location in color_point:
            self.board[location] = color
            self.checked_move.append((color, location))
            value = -self.alphabeta(opponent, -beta, -alpha, current_depth + 1)

            if value == 1:
                self.winning_move = location
            elif value == -1:
                self.opponent_winning_move = location

            if value > alpha:
                alpha = value
            self.board[location] = EMPTY
            if value >= beta:
                return beta

        return alpha

    def decide_winner(self):
        if self.winner == 1:
            #print('BLACK')
            self.winner_char = 'b'
        elif self.winner == 2:
            self.winner_char = 'w'

    def build_tree(self):
        self.checked_move = []
        self.depth = 5
        current_depth = 1
        self.best_move_for_now = {}
        for i in range(self.depth):
            self.best_move_for_now[i + 1] = []

        location = self.alphabeta(color=self.current_player,
                                  alpha=-1,
                                  beta=1,
                                  current_depth=0)
        if location > 0:
            # The location is a winning move
            self.decide_winner()
            try:
                return [self.winner_char, self.winning_move]
            except AttributeError:
                return [self.winner_char]
        elif location == 0:
            # This is a draw or the program reachs the time limit.
            opponent = opponent = WHITE + BLACK - self.current_player
            player_dict, opponent_dict = self.mapping_all_heuristic(opponent)
            player_key, player_value = max(player_dict.items(),
                                           key=lambda p: p[1])
            opponent_key, opponent_value = max(opponent_dict.items(),
                                               key=lambda p: p[1])
            if opponent_value > player_value:
                return ['draw', opponent_key]
            else:
                return ['draw', player_key]

        elif location < 0:
            # There is no wining move for player right now.
            self.decide_winner()
            return [self.winner_char]