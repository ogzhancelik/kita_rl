import torch
import numpy as np
import multiprocessing as mp
from model import ResNet
from mcts import MCTS
from env import Kita
from train import Train
import utils as f

class AlphaZero:
    def __init__(self, args):
        self.args = args
        self.model = ResNet().to(args["device"])
        self.mcts = MCTS(args, self.model)

    def game_policy(self, state: Kita) -> tuple[str, np.ndarray]:
        mcts_action_probs, depth = self.mcts.search(state)
        
        if state.move_counter < 6:
            action = np.random.choice(len(mcts_action_probs), p=mcts_action_probs)
            return f.a0_to_move(int(action)), mcts_action_probs, depth
        action = np.argmax(mcts_action_probs)
        return f.a0_to_move(int(action)), mcts_action_probs, depth

    @torch.no_grad()
    def self_play(self) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        probs = []
        states = []
        actions = []
        fm_interest = []
        depth_hist = []
        var_hist = []
        num_valid_moves = []
        state = Kita()
        
        while not state.check_gameover() and state.move_counter < 256:
            action, prob, depth = self.game_policy(state)
            fm_interest.append(max(prob))
            depth_hist.append(depth)
            var_hist.append(np.var(prob))
            num_valid_moves.append(state.num_valid_moves())
            actions.append(action)
            states.append(f.board_to_matrix(state))
            probs.append(prob)
            state.move(action)

        if state.move_counter < 256:
            states.append(f.board_to_matrix(state))
            probs.append(np.zeros((672)))

        winner = state.check_gameover()
        win_list  = [winner * (-1)**i for i in range(len(states))]
        return states, probs, win_list, fm_interest, sum(depth_hist) / len(depth_hist), state.move_counter, winner, var_hist, sum(num_valid_moves)/len(num_valid_moves)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path)["model_state_dict"])

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def play_game(args):
    A0 = AlphaZero(args)
    return A0.self_play()

import os
import csv

def save_statistics_csv(fm_interest, depth_hist, game_len, win, var_hist, num_valid_moves):
    """Her istatistiği kendi ayrı CSV dosyasına ekleyerek kaydeder."""

    def append_to_csv(file, data, header):
        """Veriyi belirtilen CSV dosyasına ekler (append modunda)."""
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

    # Her istatistiği ayrı dosyalara kaydet
    append_to_csv("fm_interest.csv", fm_interest, "fm_interest")
    append_to_csv("depth_hist.csv", depth_hist, "depth_hist")
    append_to_csv("game_len.csv", game_len, "game_len")
    append_to_csv("win.csv", win, "win")
    append_to_csv("var_hist.csv", var_hist, "var_hist")
    append_to_csv("num_valid_moves.csv", num_valid_moves, "num_valid_moves")

# Self-play tamamlandığında çağır






# Self-play tamamlandığında çağır


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = {
        "num_simulation": 50,

        "c_base": 19652,
        "c_init": 1.25,
        "dirichlet_epsilon": 0.,
        "dirichlet_alpha": 0.001,
        "action_space": 672,
        "device": device,
        "top_actions": 15,
        "t": 1,
        "step": 3,
        "model_checkpoint_path": "model_epoch_1.pth",
        "num_games": 8,  # Paralel olarak çalıştırılacak oyun sayısı
        "games_per_training": 1,  # Kaç oyun oynandıktan sonra eğitime geçileceği
        "max_cycles": 1,  # Kaç defa 1000 oyun + eğitim döngüsü yapılacağı
        # Her eğitimde kaç epoch çalıştırılacağı
    }

    A0 = AlphaZero(args)
    trainer = Train()

    for cycle in range(args["max_cycles"]):
        print(f"Cycle {cycle+1}/{args['max_cycles']} - Starting {args['games_per_training']} games before training")

        total_games = 0
        all_boards, all_moves, all_evals = [], [], []
        all_fm_interest, all_depth, all_game_len, all_win, all_var_hist, all_num_valid_moves = [], [], [], [], [], []

        # Her 1000 oyun tamamlanana kadar oyunları paralel çalıştır
        while total_games < args["games_per_training"]:
            print(f"Running {args['num_games']} parallel games...")
            
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(play_game, [args] * args["num_games"])

            for result in results:
                boards, moves, evals, fm_interest, depth, game_len, win, var_hist, num_valid_moves = result
                all_boards.extend(boards)
                all_moves.extend(moves)
                all_evals.extend(evals)
                all_fm_interest.extend(fm_interest)
                all_depth.append(depth)
                all_game_len.append(game_len)
                all_win.append(win)
                all_var_hist.extend(var_hist)
                all_num_valid_moves.append(num_valid_moves)

            total_games += args["num_games"]
            print(f"Total games played: {total_games}/{args['games_per_training']}")
        save_statistics_csv(all_fm_interest, all_depth, all_game_len, all_win, all_var_hist, all_num_valid_moves)


        print("Self-play completed, formatting data...")

        # Veriyi PyTorch tensörlerine çevir
        boards = torch.tensor(np.array(all_boards), dtype=torch.float32)
        moves = torch.tensor(np.array(all_moves), dtype=torch.float32)
        evals = torch.tensor(np.array(all_evals), dtype=torch.float32)

        print("Data formatted, starting training...")

        trainer.train(boards, evals, moves)

        # Modeli her cycle sonunda kaydet
        model_path = f"model_checkpoint_cycle_{cycle+1}.pth"
        A0.save_model(model_path)
        print(f"Model saved: {model_path}")

    print("Training completed. Exiting loop.")


# for cycle in range(args["max_cycles"]):
#     print(f"Cycle {cycle+1}/{args['max_cycles']} - Starting {args['games_per_training']} games before training")

#     total_games = 0
#     all_boards, all_moves, all_evals = [], [], []
#     all_fm_interest, all_depth, all_game_len, all_win, all_var_hist, all_num_valid_moves = [], [], [], [], [], []

#     while total_games < args["games_per_training"]:
#         print(f"Running game {total_games + 1}/{args['games_per_training']}...")
#         result = play_game(args)

#         boards, moves, evals, fm_interest, depth, game_len, win, var_hist, num_valid_moves = result
#         all_boards.extend(boards)
#         all_moves.extend(moves)
#         all_evals.extend(evals)
#         all_fm_interest.extend(fm_interest)
#         all_depth.append(depth)
#         all_game_len.append(game_len)
#         all_win.append(win)
#         all_var_hist.extend(var_hist)
#         all_num_valid_moves.append(num_valid_moves)

#         total_games += 1

#     save_statistics_csv(all_fm_interest, all_depth, all_game_len, all_win, all_var_hist, all_num_valid_moves)

#     print("Self-play completed, formatting data...")

#     boards = torch.tensor(np.array(all_boards), dtype=torch.float32)
#     moves = torch.tensor(np.array(all_moves), dtype=torch.float32)
#     evals = torch.tensor(np.array(all_evals), dtype=torch.float32)

#     print("Data formatted, starting training...")

#     trainer.epochs = args["epoch"]
#     trainer.train(boards, evals, moves)

#     # Modeli her cycle sonunda kaydet (isteğe bağlı)
#     # model_path = f"model_checkpoint_cycle_{cycle+1}.pth"
#     # A0.save_model(model_path)
#     # print(f"Model saved: {model_path}")

# print("Training completed. Exiting loop.")
