import torch 
import numpy as np
import time
import utils as f
from mcts import MCTS # ithinkbettermcts.py can be used here
from model import ResNet
from env import Kita
import copy



class AlphaZero:
    def __init__(self, args):
        self.args = args
        self.model = ResNet().to(args['device'])
        self.mcts = MCTS(args, self.model)

    def game_policy(self, state: Kita) -> tuple[str, np.ndarray]:
        """
        Determine the best move for a given board state using MCTS.
        """
        mcts_action_probs = self.mcts.search(state)
        action = np.argmax(mcts_action_probs)
        return f.a0_to_move(int(action)), mcts_action_probs
    @torch.no_grad()
    def self_play(self) -> list[Kita]:
        """
        Simulate a single self-play game, returning the list of board states.
        """
        
        states = []
        state = Kita()
        while not state.check_gameover():
            print(f"Move {state.move_counter+1}")
            action, _ = self.game_policy(state)
            state.move(action)
            print(action)
            states.append(copy.deepcopy(state))
        print(f"Game result: {state.check_gameover()}")
        return states
   

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = {
        "num_simulation":400,# paperde 800 bu .d :):):):):) .d 💀
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
    checkpoint_path = args["model_checkpoint_path"]
    t = time.time()
    results = A0.self_play()

    print(f"Elapsed time: {time.time() - t}")
    print(f"Finished self-play.")

 
  

    
