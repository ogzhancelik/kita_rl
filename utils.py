import torch
import numpy as np
from env import Kita
import os
import csv

def board_to_matrix(env: Kita) -> np.ndarray:
    matrix = np.zeros((10, 4, 7), dtype=np.float32)
    
    matrix[0, *env.pieces[env.turn][0]] = 1
    matrix[1, *env.pieces[env.turn][1]] = 1
    matrix[1, *env.pieces[env.turn][2]] = 1
    
    matrix[2, *env.pieces[-env.turn][0]] = 1
    matrix[3, *env.pieces[-env.turn][1]] = 1
    matrix[3, *env.pieces[-env.turn][2]] = 1

    
    
    if env.move_counter > 1:
        matrix[4, *env.last_moves[env.turn][0]] = 1
        matrix[4, *env.last_moves[env.turn][1]] = 1
    
    if env.move_counter > 0:
        matrix[5, *env.last_moves[-env.turn][0]] = 1
        matrix[5, *env.last_moves[-env.turn][1]] = 1

    if env.turn == -1:
        matrix = np.flip(matrix, axis=(1, 2)).copy()



    matrix[6, :, :] = (env.turn + 1)//2
    matrix[7, :, :] = float(env.move_counter)/100

    matrix[8] = env.tile_values

    matrix[9] = [
            [1, 1,     1,     1,  1,     1,     1],
            [1,  0,  0,  1,  0,  0,  1],
            [1,   0,  0,  1,  0,  0,  1],
            [1, 1, 1, 1,  1,     1,     1]
        ]
    
    return matrix

def move_to_a0(move: str) -> int:
    #print(move)
    #print(move[0])
    start_col = ord(move[0]) - 97
    start_row = int(move[1]) - 1
    end_col = ord(move[2]) - 97
    end_row = int(move[3]) - 1
    start_idx = start_col + start_row * 7

    col_diff = end_col - start_col
    row_diff = end_row - start_row

    if col_diff == 0:
        move_type_index = row_diff + 2 if row_diff > 0 else row_diff+3
    elif row_diff == 0:
        move_type_index = col_diff + 8 if col_diff > 0 else col_diff+9
    else:
        if abs(row_diff)-abs(col_diff) == 0:
            move_type_index = (row_diff+2*col_diff+3)//2+12
        elif abs(row_diff) == 2:
            move_type_index = (row_diff+col_diff+3)//2+16
        elif abs(col_diff) == 2:
            move_type_index = (row_diff+col_diff+3)//2+20

    return int(start_idx + move_type_index*28)


def a0_to_move(action: int) -> str:
    L_move = {12: (-1,-1), 13: (1,-1), 14:(-1,1), 15:(1,1),
              16: (-2,-1), 17: (-2,1), 18:(2,-1), 19:(2,1),
              20: (-1,-2), 21: (1,-2), 22:(-1,2), 23:(1,2)}
    start_idx = action % 28
    move_type_index = action // 28
    start_row = start_idx // 7
    start_col = start_idx % 7
    start_square = chr(start_col + 97) + str(start_row + 1)

    if move_type_index < 3:
        end_square = chr(start_col+97) + str(start_row -3 + move_type_index + 1)
    elif move_type_index < 6:
        end_square = chr(start_col+97) + str(start_row + move_type_index - 1)
    elif move_type_index < 9:
        end_square = chr(start_col+97 + move_type_index - 9) + str(start_row + 1)
    elif move_type_index < 12:
        end_square = chr(start_col+97 + move_type_index - 8) + str(start_row + 1)
    else:
        end_square = chr(start_col+97 + L_move[move_type_index][1]) + str(start_row + L_move[move_type_index][0] + 1)
    
    return start_square + end_square


def moves_to_a0(moves: list[str]) -> list[int]:
    return [move_to_a0(move) for move in moves]


def parallel_valid_policy(policies: np.ndarray, envs: list[Kita]) -> np.ndarray:
    valid_moves = [list(env.get_valid_moves(env.turn)) for env in envs]
    encoded_valid_moves = [moves_to_a0(moves) for moves in valid_moves]
    mask = np.zeros(shape=(len(envs), 672))
    for i, moves in enumerate(encoded_valid_moves):
        mask[i, moves] = 1
    valid_policy = mask * policies
    row_sums = np.sum(valid_policy, axis=1, keepdims=True, dtype=np.float32)
    valid_policy /=  np.where(row_sums != 0, row_sums, 1)
    return valid_policy

def valid_policy(policy: np.ndarray, env: Kita) -> np.ndarray:
    valid_moves = list(env.get_valid_moves(env.turn))
    encoded_valid_moves = moves_to_a0(valid_moves)
    mask = np.zeros(672)
    mask[encoded_valid_moves] = 1
    valid_policy = mask * policy
    valid_policy /= (np.sum(valid_policy) if np.sum(valid_policy) != 0 else 1)
    return valid_policy



def prepare_input(env: Kita) -> torch.Tensor:
    matrix = board_to_matrix(env)
    X_tensor = torch.tensor(matrix, dtype=torch.float32)
    return X_tensor

def mirror_move(move: str) -> str:
    return chr(200- ord(move[0])) + str(5-int(move[1])) + chr(200 - ord(move[2])) + str(5-int(move[3]))

def mirror_moves(moves: list[str]) -> list[str]:
    return [mirror_move(move) for move in moves]

import torch
import torch.nn.functional as F

def kl_divergence(p, q):
    return torch.sum(p * (torch.log(p) - torch.log(q)))



def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)







def append_to_csv(file, data, header):
        """Veriyi belirtilen CSV dosyasına ekler (append modunda)."""
        dir = "stats"
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        file = os.path.join(dir, file)
        file_exists = os.path.exists(file)
        with open(file, "a", newline="") as f:
            writer = csv.writer(f)

            # Eğer dosya yoksa başlık ekle
            if not file_exists:
                writer.writerow([header])

            # Yeni verileri satır satır ekle
            for value in data:
                writer.writerow([value])

        print(f"Appended to {file}")