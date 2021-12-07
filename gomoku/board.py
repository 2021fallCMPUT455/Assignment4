"""
board.py
Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move
The board uses a 1-dimensional representation with padding
"""

import numpy as np
import random
from board_util import (
    GoBoardUtil,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    PASS,
    is_black_white,
    is_black_white_empty,
    coord_to_point,
    where1d,
    MAXSIZE,
    GO_POINT
)

"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.
The board is stored as a one-dimensional array of GO_POINT in self.board.
See GoBoardUtil.coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size):
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()

    def calculate_rows_cols_diags(self):
        if self.size < 5:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)

            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)

        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 5:
                self.diags.append(diag_NE)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >=5:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 5) + 1) * 2

    def reset(self, size):
        """
        Creates a start state, an empty board with given size.
        """
        self.size = size
        self.NS = size + 1
        self.WE = 1
        self.ko_recapture = None
        self.last_move = None
        self.last2_move = None
        self.current_player = BLACK
        self.maxpoint = size * size + 3 * (size + 1)
        self.board = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()

    def copy(self):
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point):
        return self.board[point]

    def pt(self, row, col):
        return coord_to_point(row, col, self.size)

    def is_legal(self, point, color):
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        board_copy = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def get_empty_points(self):
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def get_color_points(self, color):
        """
        Return:
            All points of color on the board
        """
        return where1d(self.board == color)

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
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point, color):
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block):
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone):
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block
        """
        color = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point):
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=bool)
        pointstack = [point]
        color = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point):
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns None otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture = None
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture

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
            self.last2_move = self.last_move
            self.last_move = point
            return True
        elif self.board[point] != EMPTY:
            return False
        # if point == self.ko_recapture:
        #     return False

        # General case: deal with captures, suicide, and next ko point
        # opp_color = GoBoardUtil.opponent(color)
        # in_enemy_eye = self._is_surrounded(point, opp_color)
        self.board[point] = color
        # single_captures = []
        # neighbors = self._neighbors(point)
        # for nb in neighbors:
        #     if self.board[nb] == opp_color:
        #         single_capture = self._detect_and_process_capture(nb)
        #         if single_capture != None:
        #             single_captures.append(single_capture)
        # block = self._block_of(point)
        # if not self._has_liberty(block):  # undo suicide move
        #     self.board[point] = EMPTY
        #     return False
        # self.ko_recapture = None
        # if in_enemy_eye and len(single_captures) == 1:
        #     self.ko_recapture = single_captures[0]
        self.current_player = GoBoardUtil.opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        return True

    def neighbors_of_color(self, point, color):
        """ List of neighbors of point of given color """
        nbc = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point):
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point):
        """ List of all four diagonal neighbors of point """
        return [
            point - self.NS - 1,
            point - self.NS + 1,
            point + self.NS - 1,
            point + self.NS + 1,
        ]

    def last_board_moves(self):
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not None, not PASS).
        """
        board_moves = []
        if self.last_move != None and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != None and self.last2_move != PASS:
            board_moves.append(self.last2_move)
            return

    def detect_five_in_a_row(self):
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, list):
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY

    def undo_all_move(self, point_list):
        for point in point_list:
            self.board[point] = EMPTY
    '''
    def find_all_potential_moves(self, simulation_amount):
        print(self.current_player)
        color = self.current_player
        opponent = opponent = WHITE + BLACK - color
        potential_moves = self.get_empty_points().tolist()
        random.shuffle(potential_moves)
        move_list = 'Random'
        win_rate_dictionary = {}
        for move in potential_moves:
            move_list += (' ' + str(move))
            self.play_move(move, self.current_player)
            win = 0
            loss = 0
            for n_th in range(simulation_amount):
                winner, win_move = self.random_policy()
                
                if winner == color:
                    win += 1
                elif winner == opponent:
                    loss += 1
            self.board[move] = EMPTY
            win_rate = win / simulation_amount
            win_rate_dictionary[move] = win_rate
        best_move = max(win_rate_dictionary, key=win_rate_dictionary.get)
        #self.board[best_move] = color
        return best_move, win_rate_dictionary
    def random_policy(self):
        color = self.current_player
        opponent = WHITE + BLACK - color
        empty_points = self.get_empty_points().tolist()
        made_move_list = []
        random.shuffle(empty_points)
        current_move = 0
        winner = EMPTY
        while winner == EMPTY and len(empty_points) != 0:
            current_move = empty_points[0]
            self.play_move(current_move, self.current_player)
            
            made_move = empty_points.pop(0)
            made_move_list.append(made_move)
            winner = self.detect_five_in_a_row()
        self.undo_all_move(made_move_list)
        return winner, current_move
    '''
    def analyze_hor(self, point, color):
        opponent = WHITE + BLACK - color
        y = point % self.NS
        x = point // self.NS

        y_counter = 0

        right_neighbor = -1
        left_neighbor = -1
        dead_line = 0
        one_side_color = 0
        two_side_color = 0
        has_empty_end = 0


        for y_marker in range(y + 1, self.NS):
            color_stone_line = self.get_color(self.pt(x, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x, y_marker)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            y_counter += 1

        if y_counter != 0:
            one_side_color += y_counter

        for y_marker in range(y - 1, 0, -1):
            color_stone_line = self.get_color(self.pt(x, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x, y_marker)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            y_counter += 1
        '''
        two_side_color += y_counter
        if two_side_color > one_side_color:
            # Then, this is a empty point in the middle of a line.
            if dead_line == 2 and y_counter < 4:
                y_counter = 0
        else:
            if dead_line == 1 and y_counter < 4:
                y_counter = 0
        '''
        if has_empty_end != 2 and y_counter < 4:
            y_counter = 0
        return point, y_counter

    def analyze_ver(self, point, color):
        y = point % self.NS
        x = point // self.NS

        x_counter = 0
        right_neighbor = -1
        left_neighbor = -1
        dead_line = 0
        one_side_color = 0
        two_side_color = 0
        has_empty_end = 0

        for x_marker in range(x + 1, self.NS):
            color_stone_line = self.get_color(self.pt(x_marker, y))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x_marker, y)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            x_counter += 1

        if x_counter > 0:
            one_side_color += x_counter

        for x_marker in range(x - 1, 0, -1):
            color_stone_line = self.get_color(self.pt(x_marker, y))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x_marker, y)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            x_counter += 1
        '''
        two_side_color += x_counter
        if two_side_color > one_side_color:
            # Then, this is a empty point in the middle of a line.
            if dead_line == 2 and x_counter < 4:
                x_counter = 0
        else:
            if dead_line == 1 and x_counter < 4:
                x_counter = 0
        '''
        if has_empty_end != 2 and x_counter < 4:
            x_counter = 0
        return point, x_counter

    def analyze_left_diag(self, point, color):
        y = point % self.NS
        x = point // self.NS

        counter = 0
        right_neighbor = -1
        left_neighbor = -1
        dead_line = 0
        one_side_color = 0
        two_side_color = 0
        has_empty_end = 0

        for x_marker, y_marker in zip(range(x + 1, self.NS), range(y + 1, self.NS)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x_marker, y_marker)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            counter += 1

        if counter > 0:
            one_side_color += counter

        for x_marker, y_marker in zip(range(x - 1, 0, -1), range(y - 1, 0,
                                                                 -1)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x_marker, y_marker)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            counter += 1
        '''
        two_side_color += counter
        if two_side_color > one_side_color:
            # Then, this is a empty point in the middle of a line.
            if dead_line == 2 and counter < 4:
                counter = 0
        else:
            if dead_line == 1 and counter < 4:
                counter = 0
        '''
        if has_empty_end != 2 and counter < 4:
            counter = 0
        return point, counter

    def analyze_right_diag(self, point, color):
        y = point % self.NS
        x = point // self.NS

        counter = 0
        right_neighbor = -1
        left_neighbor = -1
        dead_line = 0
        one_side_color = 0
        two_side_color = 0
        has_empty_end = 0

        for x_marker, y_marker in zip(range(x - 1, 0, -1), range(y + 1, self.NS)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                right_neighbor = self.pt(x_marker, y_marker)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            counter += 1

        if counter > 0:
            one_side_color += counter

        for x_marker, y_marker in zip(range(x + 1, self.NS),
                                      range(y - 1, 0, -1)):
            color_stone_line = self.get_color(self.pt(x_marker, y_marker))
            if color_stone_line == color:
                pass
            elif color_stone_line == EMPTY:
                left_neighbor = self.pt(x_marker, y_marker)
                has_empty_end += 1
                break
            else:
                dead_line += 1
                break
            counter += 1
        '''
        two_side_color += counter
        if two_side_color > one_side_color:
            # Then, this is a empty point in the middle of a line.
            if dead_line == 2 and counter < 4:
                counter = 0
        else:
            if dead_line == 1 and counter < 4:
                counter = 0
        '''
        if has_empty_end != 2 and counter < 4:
            counter =0
        return point, counter



    def mapping_player_heuristic(self, color):
        '''
        player_stone_list = [point for point in where1d(self.board == color)]
        if len(player_stone_list) == 0:
            player_stone_list = [
                point for point in where1d(self.board == EMPTY)
            ]
        '''
        player_stone_list = [point for point in where1d(self.board == EMPTY)]

        potential_move_dict = {}

        for point in player_stone_list:
            potential_moves = []

            point, counter = self.analyze_hor(
                point, color)


            potential_moves.append([point, counter])
            point, counter = self.analyze_ver(
                point, color)

            potential_moves.append([point, counter])
            point, counter = self.analyze_left_diag(
                point, color)

            potential_moves.append([point, counter])
            point, counter = self.analyze_right_diag(
                point, color)

            potential_moves.append([point, counter])

            for value_pair in potential_moves:
                if value_pair[0] not in potential_move_dict.keys(
                ) and value_pair[0] != -1:
                    potential_move_dict[value_pair[0]] = value_pair[1]
                elif value_pair[0] in potential_move_dict.keys():
                    if value_pair[1] > potential_move_dict[value_pair[0]]:
                        potential_move_dict[value_pair[0]] = value_pair[1]

        #return max(potential_move_dict.items(), key=lambda k : k[1])
        return potential_move_dict

    def mapping_all_heuristic(self, color):
        opponent = WHITE + BLACK - color

        player_dict = self.mapping_player_heuristic(color)
        opponent_dict = self.mapping_player_heuristic(opponent)
        #player_best_move = max(player_dict, key=lambda k : player_dict[k])
        #opponent_best_move = max(opponent_dict, key=lambda k : opponent_dict[k])

        player_dict = dict(
            sorted(player_dict.items(), key=lambda item: item[1]))
        opponent_dict = dict(
            sorted(opponent_dict.items(), key=lambda item: item[1]))

        return player_dict, opponent_dict

    def find_all_potential_moves(self, simulation_amount):

        color = self.current_player
        opponent = opponent = WHITE + BLACK - color
        potential_moves = self.get_empty_points().tolist()
        random.shuffle(potential_moves)
        move_list = 'Random'
        win_rate_dictionary = {}
        for move in potential_moves:
            move_list += (' ' + str(move))
            #self.play_move(move, self.current_player)
            self.board[move] = color
            if self.detect_five_in_a_row() == color:
                win_rate_dictionary[move] = 1.0
            elif len(self.get_empty_points().tolist()) == 0:
                win_rate_dictionary[move] = 0
            else:
                win = 0
                loss = 0
                for n_th in range(simulation_amount):
                    winner = self.random_policy(opponent)

                    if winner == color:
                        win += 1
                    elif winner == opponent:
                        loss += 1
                #self.board[move] = EMPTY
                win_rate = win / simulation_amount
                win_rate_dictionary[move] = win_rate
            self.board[move] = EMPTY
        win_rate_dictionary = dict( sorted(win_rate_dictionary.items(),
                           key=lambda item: item[1],
                           reverse=True))
        if len(win_rate_dictionary) == 0:
            best_move = []
        else:
            best_move = max(win_rate_dictionary, key=win_rate_dictionary.get)  # if len(best_move) == 0:

        # est_move = max(win_rate_dictionary, key=win_rate_dictionary.get)        # if len(best_move) == 0:
        #     best_move = []
        # print(win_rate_dictionary)
        #self.board[best_move] = color
        return best_move

    def random_policy(self, opponent):
        color = opponent
        empty_points = self.get_empty_points().tolist()
        made_move_list = []
        random.shuffle(empty_points)
        '''
        current_move = 0
        winner = EMPTY
        while winner == EMPTY and len(empty_points) != 0:
            current_move = empty_points[0]
            self.play_move(current_move, self.current_player)
            
            made_move = empty_points.pop(0)
            made_move_list.append(made_move)
            winner = self.detect_five_in_a_row()
        '''
        winner = self.detect_five_in_a_row()
        for empty_point in empty_points:
            #self.play_move(empty_point, self.current_player)
            self.board[empty_point] = color
            made_move_list.append(empty_point)
            winner = self.detect_five_in_a_row()
            if color == BLACK:
                color = WHITE
            elif color == WHITE:
                color = BLACK
            if winner != EMPTY:
                self.undo_all_move(made_move_list)
                break
        self.undo_all_move(made_move_list)
        return winner

    def find_move_under_rule(self, simulation_amount):

        color = self.current_player
        opponent = opponent = WHITE + BLACK - color
        potential_moves = self.get_empty_points().tolist()
        random.shuffle(potential_moves)

        win_str = 'Win'
        block_win_str = 'BlockWin'
        open_four_str = 'OpenFour'
        block_open_four_str = 'BlockOpenFour'
        random_str = 'Random'

        win_list = []
        block_win_list = []
        open_four_list = []
        block_open_four_list = []
        random_list = []
        player_dict, opponent_dict = self.mapping_all_heuristic(color)


        for key, value in player_dict.items():
            if value == 4 :
                win_list.append(key)
                win_str += (' ' + str(key))
            elif value == 3 :
                open_four_list.append(key)
                open_four_str += (' ' + str(key))

        for key, value in opponent_dict.items():
            if value == 4 :
                block_win_list.append(key)
                block_win_str += (' ' + str(key))
            elif value == 3 :
                block_open_four_list.append(key)
                block_open_four_str += (' ' + str(key))

        for move in self.get_empty_points().tolist():
            if move not in win_list and move not in open_four_list and move not in  block_win_list and move not in block_open_four_list:
                random_list.append(move)
                random_str += (' ' + str(move))

        move_priority_list = win_list + block_win_list + open_four_list + block_open_four_list + random_list
        win_rate_dictionary = {}
        return win_str, block_win_str, open_four_str, block_open_four_str, random_str

        # print(win_str)
        # print(block_win_str)
        # print(open_four_str)
        # print(block_open_four_str)
        # print(random_str)
        '''
        for move in move_priority_list:
           
            #self.play_move(move, color)
            self.board[move] = color
            current_winner = self.detect_five_in_a_row()
            
            if current_winner == color:
                
                win_rate_dictionary[move] = 1.0
                
            elif current_winner != color:
                win = 0
                loss = 0
                for n_th in range(simulation_amount):
                    winner = self.random_policy(color)
                    
                    if winner == color:
                        win += 1
                    elif winner == opponent:
                        loss += 1
                
                win_rate = win / simulation_amount
                win_rate_dictionary[move] = win_rate
            self.board[move] = EMPTY
        best_move = max(win_rate_dictionary, key=win_rate_dictionary.get)
        #self.board[best_move] = color
        
        return best_move, win_rate_dictionary[best_move]
        '''