import torch
import numpy as np
import multiprocessing as mp
from model import ResNet
from mcts import MCTS
from env import Kita
from train import Train
import utils as f
import os
import csv

class AlphaZero:
    def __init__(self, args):
        self.args = args
        self.model = ResNet().to(args["device"])
        self.mcts = MCTS(args, self.model)
        self.device = args["device"]
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scaler = torch.amp.GradScaler()  # AMP için GradScaler
 

    def game_policy(self, state: Kita, random = True) -> tuple[str, np.ndarray]:
        mcts_action_probs, depth = self.mcts.search(state)
        
        if random and (state.move_counter < 10):
            action = np.random.choice(len(mcts_action_probs), p=mcts_action_probs)
            return f.a0_to_move(int(action)), mcts_action_probs, depth
        action = np.argmax(mcts_action_probs)
        return f.a0_to_move(int(action)), mcts_action_probs, depth


    def play_vs(self, model_path):

        self.load_model(model_path)
        state = Kita()

        while not state.check_gameover():
            if state.move_counter % 2 == 0:
                action, _, _ = self.game_policy(state, random=False)
                print(f"AI move: {action}")
            else:
                action = input("Enter move: ")
            state.move(action)
            
        
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
        self.model.load_state_dict(torch.load(path))

    def load_model_inference(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()



    def load_model_train(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])  
        return checkpoint['epoch']  # Kaldığın epoch'u döndür


    def save_model(self, path, cycle):
        torch.save({
            'cycle': cycle,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # AMP için ekledik
        }, path)



def play_game(args):
    A0 = AlphaZero(args)
    return A0.self_play()


def save_statistics_csv(fm_interest, depth_hist, game_len, win, var_hist, num_valid_moves):
    """Her istatistiği kendi ayrı CSV dosyasına ekleyerek kaydeder."""

    # Her istatistiği ayrı dosyalara kaydet
    f.append_to_csv("fm_interest.csv", fm_interest, "fm_interest")
    f.append_to_csv("depth_hist.csv", depth_hist, "depth_hist")
    f.append_to_csv("game_len.csv", game_len, "game_len")
    f.append_to_csv("win.csv", win, "win")
    f.append_to_csv("var_hist.csv", var_hist, "var_hist")
    f.append_to_csv("num_valid_moves.csv", num_valid_moves, "num_valid_moves")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = {
        "num_simulation": 400,
        "c_base": 19652,
        "c_init": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 2.5,
        "action_space": 672,
        "device": device,
        "top_actions": 15,
        "t": 1,
        "step": 3,
        "model_checkpoint_path": "model_epoch_1.pth",
        "num_games": 8,  # Paralel olarak çalıştırılacak oyun sayısı
        "games_per_training": 128,  # Kaç oyun oynandıktan sonra eğitime geçileceği
        "max_cycles": 10,  
    }

    A0 = AlphaZero(args)
    A0.play_vs(r"./models/checkpoint_cycle_20.pth")

    # trainer = Train()

    # for cycle in range(args["max_cycles"]):
    #     print(f"Cycle {cycle+1}/{args['max_cycles']} - Starting {args['games_per_training']} games before training")

    #     total_games = 0
    #     all_boards, all_moves, all_evals = [], [], []
    #     all_fm_interest, all_depth, all_game_len, all_win, all_var_hist, all_num_valid_moves = [], [], [], [], [], []

    #     while total_games < args["games_per_training"]:
    #         print(f"Running {args['num_games']} parallel games...")
            
    #         with mp.Pool(processes=mp.cpu_count()) as pool:
    #             results = pool.map(play_game, [args] * args["num_games"])

    #         for result in results:
    #             boards, moves, evals, fm_interest, depth, game_len, win, var_hist, num_valid_moves = result
    #             all_boards.extend(boards)
    #             all_moves.extend(moves)
    #             all_evals.extend(evals)
    #             all_fm_interest.extend(fm_interest)
    #             all_depth.append(depth)
    #             all_game_len.append(game_len)
    #             all_win.append(win)
    #             all_var_hist.extend(var_hist)
    #             all_num_valid_moves.append(num_valid_moves)

    #         total_games += args["num_games"]
    #         print(f"Total games played: {total_games}/{args['games_per_training']}")
    #     save_statistics_csv(all_fm_interest, all_depth, all_game_len, all_win, all_var_hist, all_num_valid_moves)


    #     print("Self-play completed, formatting data...")

    #     # Veriyi PyTorch tensörlerine çevir
    #     boards = torch.tensor(np.array(all_boards), dtype=torch.float32)
    #     moves = torch.tensor(np.array(all_moves), dtype=torch.float32)
    #     evals = torch.tensor(np.array(all_evals), dtype=torch.float32)

    #     print("Data formatted, starting training...")

    #     trainer.train(boards, evals, moves)

    #     # Modeli her cycle sonunda kaydet
    #     model_path = f"model_checkpoint_cycle_{cycle+1}.pth"
    #     A0.save_model(model_path, cycle+1)
    #     print(f"Model saved: {model_path}")

    # print("Training completed. Exiting loop.")
