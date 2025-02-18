from collections import deque

class BoardGameEnv:
    def __init__(self):
        self.board = [
            [-2, -1,  0,  0,  0,  0,  0],
            [-1,  None,  None,  0,  None,  None,  0],
            [0,  None,  None,  0,  None,  None,  1],
            [0,  0,  0,  0,  0,  1,  2]
        ]
        self.tile_values = [
            [2, 3, 1, 2, 1, 3, 2],
            [3, 0, 0, 1, 0, 0, 3],
            [3, 0, 0, 1, 0, 0, 3],
            [2, 3, 1, 2, 1, 3, 2]
        ]
        self.current_pieces = [(0,0), (0,1), (1,0)] #-2, -1, -1
        self.other_pieces = [(3,6), (3,5), (2,6)] #2, 1, 1
        self.player_turn = 1  # Player 1 starts
        self.step_size = 2
        self.last_moves = {1: (0,0,0,0), 2: (0,0,0,0)}  # Track last move per player
        self.forced_move = False
    
    def update_step_size(self):
        for row in range(4):
            for col in range(7):
                if (self.board[row][col] == 2 and self.player_turn == 1) or (self.board[row][col] == -2 and self.player_turn == 2):
                    self.step_size = self.tile_values[row][col]
                    return

    def get_valid_moves(self):
        valid_moves = []
        for (row, col) in self.current_pieces:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            queue = deque([(row, col, 0)])
            visited = set()
            valid_moves = []
            last_move = self.last_moves[self.player_turn]
            
            while queue:
                r, c, steps = queue.popleft()
                if steps == self.step_size:
                    if self.board[r][c] in [0, -2, 2] and ((r, c, row, col) != last_move or self.forced_move):
                        valid_moves.append((r, c))
                    continue
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 4 and 0 <= nc < 7 and self.board[nr][nc] is not None and (nr, nc) not in visited:
                        if self.board[nr][nc] not in [-1, 1] and not (self.board[nr][nc] in [-2, 2] and steps + 1 < self.step_size):
                            queue.append((nr, nc, steps + 1))
                            visited.add((nr, nc))
        return valid_moves

    def get_valid_move_count(self):
        self.forced_move = True
        move_count = 0
        for (row, col) in self.current_pieces:
            move_count += len(self.get_valid_moves(row, col))
        
        self.forced_move = False
        return move_count     

    def is_forced(self):
        self.forced_move = self.get_valid_move_count == 1

    def swap_pieces(self):
        c = self.current_pieces
        self.current_pieces = self.other_pieces
        self.other_pieces = c

    def move_piece(self, move): #a0b1 şeklinde
        rows_dict = {'a':0, 'b':1, 'c':2, 'd':3}
        start_row = rows_dict[move[0]]
        start_col = int(move[1])
        end_row = rows_dict[move[2]]
        end_col = int(move[3])
        if self.board[start_row][start_col] is None or self.board[end_row][end_col] is None:
            return False
        if (self.board[start_row][start_col] < 0 and self.player_turn == 1) or (self.board[start_row][start_col] > 0 and self.player_turn == 2):
            valid_moves = self.get_valid_moves(start_row, start_col)
            if (end_row, end_col) in valid_moves:
                self.board[end_row][end_col] = self.board[start_row][start_col]
                self.board[start_row][start_col] = 0
                self.last_moves[self.player_turn] = (start_row, start_col, end_row, end_col)  # Store move to prevent undo
                self.player_turn = 3 - self.player_turn  # Switch turn
                self.update_step_size()

                if (end_row,end_col) == self.other_pieces[0]:
                    self.other_pieces[0] = (-1, -1)

                for i in range(len(self.current_pieces)):
                    if self.current_pieces[i] == (start_row, start_col):
                        self.current_pieces[i] = (end_row, end_col)
                self.swap_pieces()
                self.is_forced()
                return True
        return False
    
    def check_game_over(self):
        if self.current_pieces[0] == (-1,-1):
            return True
        
        return self.get_valid_move_count() == 1

    def print_board_state(self):
        for row in self.board:
            print(" ".join(f"{'' if cell is None else cell:>3}" for cell in row))


if __name__ == "__main__":
    env = BoardGameEnv()
    env.move_piece("a1a3")
    env.print_board_state()