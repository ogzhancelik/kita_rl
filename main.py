import torch
import numpy as np
import time
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
        """MCTS ile en iyi hamleyi bul."""
        mcts_action_probs = self.mcts.search(state)
        
        if state.move_counter < 6:
            action = np.random.choice(len(mcts_action_probs), p=mcts_action_probs)
            return f.a0_to_move(int(action)), mcts_action_probs
        action = np.argmax(mcts_action_probs)
        return f.a0_to_move(int(action)), mcts_action_probs

    @torch.no_grad()
    def self_play(self) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Tek bir oyunu oynayıp veriyi döndür."""
        probs = []
        states = []
        actions = []
        state = Kita()
        
        while not state.check_gameover() and state.move_counter < 256:
            action, prob = self.game_policy(state)
            actions.append(action)
            states.append(f.board_to_matrix(state))
            probs.append(prob)
            state.move(action)
            print(f"Move {state.move_counter}, {action}")

        if state.move_counter<256:
            states.append(f.board_to_matrix(state))
            probs.append(np.zeros((672)))

        with open("games.txt", "a") as file:  # Open file in append mode
            file.write(f"{actions}\n")

        winner = state.check_gameover()
        return states, probs, [winner*((-1)**i) for i in range(len(states))]  # Kazananı her tahtaya ekle

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path)["model_state_dict"])
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def play_game(args):
    """Tek bir oyunu oynayıp sonuçları döndürmek için kullanılacak fonksiyon (multiprocessing için)."""
    A0 = AlphaZero(args)
    return A0.self_play()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = {
        "num_simulation": 50,
        "truncation": 50,
        "c_base": 19652,
        "c_init": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 2.5,
        "memory_size": 1000,
        "action_space": 672,
        "device": device,
        "top_actions": 5,
        "t": 1,
        "model_checkpoint_path": "model_epoch_1.pth",
    }
    
    trainer = Train()
    
    game_n = 0
    while True:
        num_games = 1
        game_n+=num_games
        print(f"Game {game_n}")
        all_boards = []
        all_moves = []
        all_evals = []

        print("Starting parallel self-play...")

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(play_game, [args] * num_games)

        # Sonuçları birleştir
        for boards, moves, evals in results:
            all_boards.extend(boards)
            all_moves.extend(moves)
            all_evals.extend(evals)

        print(f"Self-play completed. Games: {num_games}")
        print(f"Boards: {len(all_boards)}, Evals: {len(all_evals)}, Moves: {len(all_moves)}")

        # Veriyi PyTorch tensörlerine çevir
        boards = torch.tensor(np.array(all_boards), dtype=torch.float32)
        moves = torch.tensor(np.array(all_moves), dtype=torch.float32)
        evals = torch.tensor(np.array(all_evals), dtype=torch.float32)

        print("Data formatted, starting training...")

        # 1000 epoch boyunca eğit
        trainer.epochs = 1000
        trainer.train(boards, evals, moves)

        print("Training completed. Restarting loop...")

"""
import torch 
import numpy as np
import time
import utils as f
from mcts import MCTS # ithinkbettermcts.py can be used here
from model import ResNet
from env import Kita
import copy
from train import Train
from graphviz import Digraph


class AlphaZero:
    def __init__(self, args):
        self.args = args
        self.model = ResNet().to(args['device'])
        self.mcts = MCTS(args, self.model)

    def game_policy(self, state: Kita) -> tuple[str, np.ndarray]:

        mcts_action_probs = self.mcts.search(state)
        action = np.argmax(mcts_action_probs)
        return f.a0_to_move(int(action)), mcts_action_probs
    @torch.no_grad()
    def self_play(self) -> list[list[Kita], list[np.ndarray], int]:

        probs = []
        states = []
        state = Kita()
        while not state.check_gameover():
            print(f"Move {state.move_counter+1}")
            action, prob = self.game_policy(state)
            states.append(f.board_to_matrix(state))
            probs.append(prob)
            state.move(action)

        print(f"Game result: {state.check_gameover()}")
        return states, probs, state.check_gameover()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = {
        "num_simulation": 400,
        "truncation": 50,
        "c_base": 19652,
        "c_init": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.03,
        "memory_size": 1000,
        "action_space": 672,
        "device": device,
        "top_actions": 5,
        "t": 1,
        "model_checkpoint_path": "model_epoch_1.pth",
    }

    A0 = AlphaZero(args)
    trainer = Train()  # Eğitici sınıfını başlat

    while True:  # Sonsuz döngü
        all_boards = []
        all_moves = []
        all_evals = []

        print("Starting self-play...")
        for i in range(1):  # 100 oyun oyna
            results, probs, win = A0.self_play()
            all_boards.extend(results)
            all_moves.extend(probs)
            all_evals.extend([win] * len(results))  # Kazananı tüm tahta durumlarına ekle

            print(f"Game {i+1}/100 completed.")
        print(f"Boards: {len(all_boards)}, Evals: {len(all_evals)}, Moves: {len(all_moves)}")

        print("Self-play completed, formatting data...")

        # Veriyi uygun formata çevir
        boards = torch.tensor(np.array(all_boards), dtype=torch.float32)
        moves = torch.tensor(np.array(all_moves), dtype=torch.float32)
        evals = torch.tensor(np.array(all_evals), dtype=torch.float32)

        print("Data formatted, starting training...")

        trainer.epochs = 100  # 1000 epoch boyunca eğit
        trainer.train(boards, evals, moves)

        print("Training completed. Restarting loop...")"""