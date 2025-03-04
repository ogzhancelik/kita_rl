import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import utils as f
from model import ResNet
from dataset import ChessDataset
from torch.utils.data import DataLoader
from tqdm import tqdm # type: ignore
import json


with open('train_config.json') as file:
    args = json.load(file)
class Train:
    def __init__(self: 'Train') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet().to(self.device)
        self.step = args['step']
        self.args = args
        self.lr = self.args['learning_rate']
        self.l2_weight = self.args['l2_weight']
        self.epochs = self.args['epochs']
        self.batch_size = self.args['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight)


        self.max_games = self.args['max_games']


    def train(self: 'Train', boards: torch.Tensor, evals: torch.Tensor, moves: torch.Tensor) -> None:

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        checkpoint_dir = './checkpoints'

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Data preparation
        train_dataset = ChessDataset(boards,evals,moves)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        
        # Model, loss functions, optimizer
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()


        loss_history = []
        mse_loss_history = []
        ce_loss_history = []
        iters = 0
        for epoch in range(self.epochs):
            lr = self.optimizer.param_groups[0]['lr']
            self.model.train()
            for data, labels_mse, labels_ce in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                data = data.to(self.device)
                labels_mse = labels_mse.to(self.device)  
                labels_ce = labels_ce.to(self.device) 
                self.optimizer.zero_grad()
            
                outputs_mse, outputs_ce = self.model(data)

                loss_mse = criterion_mse(outputs_mse, labels_mse)   
                loss_ce = criterion_ce(outputs_ce, labels_ce) 
                
                loss = loss_mse + loss_ce

                loss.backward()
                self.optimizer.step()
                
               
                
                if iters % step == 0:
                    loss_history.append(loss.item())
                    mse_loss_history.append(loss_mse.item())
                    ce_loss_history.append(loss_ce.item())
                iters += 1





            if train_loss_history[-1] == min(train_loss_history):
                for file in os.listdir(checkpoint_dir):
                    if file.startswith('model_epoch_'):
                        os.remove(os.path.join(checkpoint_dir, file))
                e = epoch + 1
                print("Best model so far. Saving...")
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{e}.pth')
                torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss,
                'mse_loss': epoch_mse,
                'ce_loss': epoch_ce,
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
                
            else:
                lr *= 1
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{e}.pth')
                self.model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
                self.optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer_state_dict"])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print("Learning rate decreased to", param_group['lr'])
                print("Best model loaded")

            

        analytics_path = os.path.join(checkpoint_dir, 'training_analytics.pth')
        torch.save({
            'train_loss_history': train_loss_history,
        }, analytics_path)
        print(f"Training analytics saved at {analytics_path}")
        print("Training finished!")

    def predict(self, board, move_counter):
        self.model.eval()
        state = f.prepare_input(board, move_counter).unsqueeze(0).to(self.args['device'])
        value = self.model(state)
        return value.cpu().item()
    def main(self):
        boards,evals,moves = self.data_preparation()
        self.train(boards,evals,moves)

    

    
if __name__ == '__main__':
    train = Train()
    train.main()

