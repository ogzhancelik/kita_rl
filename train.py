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
import time


with open(r'D:\Emin\PythonProjects\ozi\kita_rl\train_config.json') as file:
    args = json.load(file)


class Train:
    def __init__(self: 'Train') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet().to(self.device)
        self.args = args
        self.lr = self.args['learning_rate']
        self.l2_weight = self.args['l2_weight']
        self.epochs = self.args['epochs']
        self.batch_size = self.args['batch_size']
        self.step = self.args['step']  # 1 epoch içinde kaç kez log alacağımız
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        self.scaler = torch.amp.GradScaler()  # AMP için GradScaler

    def train(self: 'Train', boards: torch.Tensor, evals: torch.Tensor, moves: torch.Tensor) -> None:
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_dataset = ChessDataset(boards, evals, moves)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()

        loss_history = []
        mse_loss_history = []
        ce_loss_history = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            epoch_mse = 0
            epoch_ce = 0

            total_iters = len(train_loader)  # 1 epoch'taki toplam iterasyon sayısı
            log_intervals = max(1, total_iters // self.step)  # Log alma sıklığı

            for batch_idx, (data, labels_mse, labels_ce) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
            ):
                data, labels_mse, labels_ce = data.to(self.device), labels_mse.to(self.device), labels_ce.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda'):
                    outputs_mse, outputs_ce = self.model(data)
                    loss_mse = criterion_mse(outputs_mse, labels_mse.view(-1, 1))
                    loss_ce = criterion_ce(outputs_ce, labels_ce)
                    loss = loss_mse + loss_ce

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                epoch_mse += loss_mse.item()
                epoch_ce += loss_ce.item()

                # # Log kaydı epoch içinde `step` kadar eşit aralıklarla alınsın
                # if batch_idx % log_intervals == 0:
                #     loss_history.append(loss.item())
                #     mse_loss_history.append(loss_mse.item())
                #     ce_loss_history.append(loss_ce.item())
                #     print(f"Epoch {epoch+1}/{self.epochs} - Batch {batch_idx+1}/{total_iters} - Loss: {loss.item():.4f} - MSE Loss: {loss_mse.item():.4f} - CE Loss: {loss_ce.item():.4f}")

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss/total_iters:.4f} - MSE Loss: {epoch_mse/total_iters:.4f} - CE Loss: {epoch_ce/total_iters:.4f}")
            # write loss to csv
            with open('loss.csv', 'a') as f:
                f.write(f"{epoch+1},{epoch_loss/total_iters},{epoch_mse/total_iters},{epoch_ce/total_iters}\n")
            

        # Eğitim istatistiklerini kaydet
        # analytics_path = os.path.join(checkpoint_dir, 'training_analytics.pth')
        # torch.save({
        #     'train_loss_history': loss_history,
        #     'mse_loss_history': mse_loss_history,
        #     'ce_loss_history': ce_loss_history,
        # }, analytics_path)
        # print(f"Training analytics saved at {analytics_path}")
        # print("Training finished!")


    def predict(self, board, move_counter):
        self.model.eval()
        state = f.prepare_input(board, move_counter).unsqueeze(0).to(self.args['device'])
        value = self.model(state)
        return value.cpu().item()


