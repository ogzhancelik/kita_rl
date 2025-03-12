from collections import deque

class Kita:
    def __init__(self):
        self.board = [
            [-2, -1,     0,     0,  0,     0,     0],
            [-1,  None,  None,  0,  None,  None,  0],
            [0,   None,  None,  0,  None,  None,  1],
            [0,   0,     0,     0,  0,     1,     2]
        ]

        self.tile_values = [
            [2, 3, 1, 2, 1, 3, 2],
            [3, 0, 0, 1, 0, 0, 3],
            [3, 0, 0, 1, 0, 0, 3],
            [2, 3, 1, 2, 1, 3, 2]
        ]

        self.turn = 1
        self.pieces = {-1:[(0,0), (0,1), (1,0)], 1:[(3,6), (3,5), (2,6)]  }
        self.current_step_size = {-1:2, 1:2}
        self.last_moves = {1: [(0,0), (0,0)], -1: [(0,0), (0,0)]}
        self.move_counter = 0
        self.forced_move = True

    def parse_move(self, moves):
        new_moves = []
        for move in moves:
            new_moves.append(chr(move[0][1]+97) + str(4-move[0][0]) + chr(move[1][1]+97) + str(4-move[1][0]))
        return new_moves


    def get_valid_moves(self, turn=0):
        if turn == 0:
            turn = self.turn
        valid_moves = []
        for (row, col) in self.pieces[turn]:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            queue = deque([(row, col, 0)])
            visited = set()
            
            while queue:
                r, c, steps = queue.popleft()
                if steps == self.current_step_size[turn]:
                    if self.board[r][c] in [0, -2, 2]:
                        if self.board[r][c]*turn <= 0:
                            valid_moves.append([(row, col), (r, c)])
                    continue
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 4 and 0 <= nc < 7 and self.board[nr][nc] is not None and (nr, nc) not in visited:
                        if self.board[nr][nc] not in [-1, 1] and not (self.board[nr][nc] in [-2, 2] and steps + 1 < self.current_step_size[turn]):
                            queue.append((nr, nc, steps + 1))
                            visited.add((nr, nc))

        if len(valid_moves) > 1 and self.last_moves[turn] in valid_moves:
                valid_moves.remove(self.last_moves[turn])

        return self.parse_move(valid_moves)
                
    def num_valid_moves(self):
        return len(self.get_valid_moves())
    
    def move(self, move):
        if len(move) > 4:
            return False
        col_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
        start_col = col_dict[move[0]]
        start_row = 4 - int(move[1])
        end_col = col_dict[move[2]]
        end_row = 4 - int(move[3])

        valids = self.get_valid_moves(self.turn)

        if move in valids:
            self.board[end_row][end_col] = self.board[start_row][start_col]
            self.board[start_row][start_col] = 0
            self.last_moves[self.turn] = [(end_row, end_col), (start_row, start_col)]
            self.move_counter+=1
            for i in range(3):
                if (start_row, start_col) == self.pieces[self.turn][i]:
                    self.pieces[self.turn][i] = (end_row, end_col)
                    break
                else:
                    continue
        else:
            print("invalid move")
            return
        
        self.change_turn()


    def change_turn(self):
        (r, c) = self.pieces[self.turn][0]
        self.turn *= -1
        self.current_step_size[self.turn] = self.tile_values[r][c]

    def check_gameover(self):
        if self.board[self.pieces[-1][0][0]][self.pieces[-1][0][1]] != -2 or len(self.get_valid_moves(-1)) == 0:
            return 1
        if self.board[self.pieces[1][0][0]][self.pieces[1][0][1]] != 2 or len(self.get_valid_moves(1)) == 0:
            return -1
        return 0

    
    def print_board_state(self):
        print()
        for i, row in enumerate(self.board):  
            print(f"{4-i}|" + " ".join(f"{'' if cell is None else cell:>3}" for cell in row))
        print("   " + "_"*27)
        print("    a   b   c   d   e   f   g")
