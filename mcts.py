import math
import utils as f
import numpy as np
import torch
from env import Kita
from model import ResNet
import copy
from graphviz import Digraph

global max_depth 
max_depth = set()

class Node:
    def __init__(self, args:'dict', state:'Kita', depth:'int'=0, parent:'Node | None'=None, action:'int|None'=None, prior:'float|None'=None, policy:'np.ndarray|None'=None):
        self.args = args
        self.state = state
        self.parent = parent
        self.prior = prior
        self.policy = policy
        self.action = action
        self.depth = depth
        
        

        self.Q = 0.
        self.N = 0
        self.children:list[Node] = []

    def is_terminal(self) -> bool:
        return self.state.check_gameover() != 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def ucb(self, child: 'Node') -> float:
        c_base = self.args["c_base"]
        c_init = self.args["c_init"]
        t = self.args["t"]
        return ((child.Q + 1e-6) / (child.N + 1e-12)) +(math.log((1 + self.N + c_base) / c_base) + c_init) * (child.prior** (1/t)) * (math.sqrt(self.N) / (1 + child.N))


    def best_child(self) -> 'Node':
        child_ucbs = [self.ucb(child) for child in self.children]
        return self.children[np.argmax(child_ucbs)]
    
    @torch.no_grad()
    def value_policy_batch(self, states: 'list[Kita]', model: 'ResNet') -> tuple[np.ndarray, np.ndarray]:

        model.eval()
        inputs = torch.stack([f.prepare_input(state) for state in states]).to(self.args['device'])
        values, policies = model(inputs)
        values = values.cpu().numpy()
        policies = torch.softmax(policies, dim=1).cpu().numpy()
        
        
        valid_policies = f.parallel_valid_policy(policies, states) 
        
        return values, valid_policies

    def expand(self, model: 'ResNet') -> None:
        top_actions = self.args['top_actions']
        new_states = []
        actions = []
        probs = []
        top_actions_probs = sorted(enumerate(self.policy), key=lambda x: x[1], reverse=True)[:top_actions]
        for action, prob in top_actions_probs:
            if prob > 0:
                new_state = copy.deepcopy(self.state)
                move = f.a0_to_move(action)
                new_state.move(move)
                new_states.append(new_state)
                actions.append(action)
                probs.append(prob)
                
                
        values,policies = self.value_policy_batch(new_states, model)
        
        max_depth.add(self.depth+1)
        for i, state in enumerate(new_states):
            if state.check_gameover() != 0:
                values[i] = -1
            child = Node(self.args, new_states[i], self.depth+1, self, actions[i], probs[i], policies[i])
            self.children.append(child)
            #child.backpropagation(values[i])
            child.backpropagation(-values[i])
     

    def backpropagation(self, rollout_value: float) -> None:
        self.N += 1

        self.Q += rollout_value
        node = self.parent
        while node is not None:
            rollout_value *= -1
            node.N += 1
            node.Q += rollout_value
            node = node.parent

class MCTS:
    def __init__(self, args: 'dict', model: 'ResNet') -> None:
        self.args = args
        self.model = model.to(args['device'])


    @torch.no_grad()
    def value_policy(self, state: 'Kita', validate: 'bool|None'=True) -> tuple[float, np.ndarray]:
       
        self.model.eval()
        value , policy = self.model(f.prepare_input(state).unsqueeze(0).to(self.args['device']))
        
        
        #value = value.cpu().item()
        policy = torch.softmax(policy.squeeze(0), dim=0).cpu().numpy()
       
        if validate:
            valid_policy = f.valid_policy(policy, state)
        return value, valid_policy




    def _simulate(self, root: 'Node') -> None:
        current_node = root
        while not current_node.is_leaf():
            current_node = current_node.best_child()

        if current_node.is_terminal():
            value = 1 #if current_node.state.turn == 1 else 0
            current_node.backpropagation(value)
        else:
            current_node.expand(self.model)


    def plot_tree(self, node, filename="mcts_tree"):
        """
        Plot the MCTS tree using Graphviz.

        Parameters:
        -----------
        node : Node
            The root node of the tree to be plotted.
        filename : str
            The name of the output file (without extension).
        """
        dot = Digraph(comment='MCTS Tree')

        def add_node(dot, node, parent_id=None):
            node_id = str(id(node))
            #label = f"{node.action}\nN: {node.N}\nD: {node.depth:.2f}\nQ: {node.Q:.2f}\n"
            #label = f"{f.a0_to_move(node.action)}\nN: {node.N}"
            action = f.a0_to_move(node.action) if node.action else "None"
            label = f"{action}\nN: {node.N}\nD: {node.depth}\nQ: {node.Q[0]:.2f}"
            fillcolor = "white"
            if node.is_terminal():
                if node.depth%2 == 0:
                    fillcolor = "lightcoral"
                else:
                    fillcolor = "darkolivegreen2"
            dot.node(node_id, label=label, style="filled", fillcolor=fillcolor)

            if parent_id is not None:
                dot.edge(parent_id, node_id)

            for child in node.children:
                add_node(dot, child, node_id)

        add_node(dot, node)

        # Render the graph to a file
        dot.render(filename, format='svg')
        print(f"Tree saved to {filename}.svg")

    def search(self, state: 'Kita') -> np.ndarray:
        max_depth.clear()
        root_state = copy.deepcopy(state)
        value, policy = self.value_policy(root_state,validate=False)
     
        
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet(
            [self.args['dirichlet_alpha']] * self.args['action_space'])
        
        policy = f.valid_policy(policy, root_state)
        root = Node(self.args, root_state)
        root.policy = policy

        for _ in range(self.args["num_simulation"]):
            self._simulate(root)
        print("max_depth", max(max_depth))

        if state.move_counter == 0:
            self.plot_tree(root, filename=f"mcts_tree")
    
        mcts_action_probs = np.zeros(self.args['action_space'])
        for child in root.children:
            mcts_action_probs[child.action] = child.N
        return mcts_action_probs / np.sum(mcts_action_probs), max(max_depth)
        #softmax
