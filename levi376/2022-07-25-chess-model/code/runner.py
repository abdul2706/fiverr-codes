"""
Original file is located at
https://colab.research.google.com/drive/1d9oBD1JE3hI3TeIYlYJIycrsxOL6Tljs
"""

from models import *

# general libraries
import os
import numpy as np
from datetime import datetime

# deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter

# this variable will help using gpu if it's available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device:', device)

"""# Define Runner"""

class Runner:
    
    RUNNER_TYPES = {'CLS': 'CLS', 'REG': 'REG'}

    def __init__(self, runner_type='REG', work_dir='', 
                 model=ChessNetRegression(), 
                 criterion=nn.CrossEntropyLoss(), 
                 optimizer=optim.SGD, lr=0.01, 
                 save_weight_file='ChessNet.pth', load_weight_file='ChessNet.pth'):
        assert runner_type in self.RUNNER_TYPES

        self.runner_type = runner_type
        self.work_dir = work_dir
        self.models_dir = os.path.join(self.work_dir, 'models')
        if not os.path.exists(self.work_dir): os.makedirs(self.work_dir)
        if not os.path.exists(self.models_dir): os.mkdir(self.models_dir)

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

        self.model_save_path = os.path.join(self.models_dir, save_weight_file)
        self.model_load_path = os.path.join(self.models_dir, load_weight_file)
        if os.path.exists(self.model_load_path):
            self.model.load_state_dict(torch.load(self.model_load_path), strict=False)
            print('loaded weights from', self.model_load_path)

    def train(self, train_loader, val_loader=None, epochs=100, log_interval=1000, writer_name=None):
        writer_name = writer_name if writer_name else datetime.now().strftime('%y-%m-%d-%H-%M')
        self.writer_path = os.path.join(self.work_dir, 'runs', writer_name)
        if not os.path.exists(self.writer_path): os.makedirs(self.writer_path)
        self.writer = SummaryWriter(self.writer_path)

        print('Training iterations in one Epoch:', len(train_loader))
        if val_loader:
            print('Validation iterations in one Epoch:', len(val_loader))

        self.total_steps = {'Train': 0, 'Val': 0}

        # loop over the dataset multiple times
        for epoch in range(epochs):

            # training part
            self.model.train()
            train_losses, train_y_true, train_y_pred = self.run_epoch(train_loader, epoch, log_interval, 'Train')
            # log epoch statistics
            log_string = f'[Train][{epoch + 1}/{epochs}] '
            log_string += f'loss: {np.mean(train_losses):.4f}, '
            log_string += f'loss_min: {np.min(train_losses):.4f}, '
            log_string += f'loss_max: {np.max(train_losses):.4f}, '
            log_string += f'loss_std: {np.std(train_losses):.4f}'
            print(log_string)
            self.writer.add_scalar(f'Epoch-Loss/Train', np.mean(train_losses), epoch)

            # save model checkpoint
            torch.save(self.model.state_dict(), self.model_save_path)
            print('saved weights at', self.model_save_path)
            self.scheduler.step()
            self.writer.add_scalar(f'Learning-Rate', self.scheduler.get_lr()[0], epoch)

            # validation part
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_losses, val_y_true, val_y_pred = self.run_epoch(val_loader, epoch, log_interval, 'Val')
                    val_loss = np.mean(val_losses)
                    log_string = f'[Val][{epoch + 1}/{epochs}] '
                    log_string += f'loss: {val_loss:.4f}'
                    print(log_string)
                    self.writer.add_scalar(f'Epoch-Loss/Val', val_loss, epoch)

    def run_epoch(self, dataloader, epoch, log_interval, run_type='Train'):
        epoch_losses = []
        epoch_y_true = []
        epoch_y_pred = []

        # iterate over all dataloader, one batch at a time
        for i, (inputs, y_true) in enumerate(dataloader):
            # pre-processing
            if self.runner_type == self.RUNNER_TYPES['CLS']:
                inputs, y_true = inputs.to(dtype=torch.float32, device=device), y_true.to(device=device)
                y_true = y_true + 128
            elif self.runner_type == self.RUNNER_TYPES['REG']:
                inputs, y_true = inputs.to(dtype=torch.float32, device=device), y_true.unsqueeze(1).to(dtype=torch.float32, device=device)
            else:
                raise Exception('Runner Error: runner_type undefined')
            if isinstance(self.model.nn_layers[0], nn.Conv1d):
                inputs = inputs.unsqueeze(1)

            if run_type == 'Train':
                # zero the parameter gradients
                self.optimizer.zero_grad()

            # forward pass
            y_pred = self.model(inputs)
            # calculate loss
            loss = self.criterion(y_pred, y_true)

            if run_type == 'Train':
                # backward pass
                loss.backward()
                # optimize weights
                self.optimizer.step()

            # post-processing
            epoch_losses.append(loss.item())
            epoch_y_true += y_true.tolist()
            epoch_y_pred += y_pred.tolist()

            if i > 0 and log_interval > 0 and i % log_interval == 0:
                # log statistics
                interval_losses = epoch_losses[-log_interval:]
                log_string = f'[{run_type}][{epoch + 1}, {i + 1:>5}] '
                log_string += f'loss: {np.mean(interval_losses):.4f}, '
                log_string += f'loss_min: {np.min(interval_losses):.4f}, '
                log_string += f'loss_max: {np.max(interval_losses):.4f}, '
                log_string += f'loss_std: {np.std(interval_losses):.4f}'
                print(log_string)
                self.total_steps[run_type] += log_interval
                self.writer.add_scalar(f'Loss/{run_type}', np.mean(interval_losses), self.total_steps[run_type])

        return epoch_losses, epoch_y_true, epoch_y_pred

    def test(self, dataloader):
        self.model.eval()
        test_losses = []
        test_y_true = []
        test_y_pred = []

        # iterate over all dataloader, one batch at a time
        for i, (inputs, y_true) in enumerate(dataloader):
            # pre-processing
            if self.runner_type == self.RUNNER_TYPES['CLS']:
                inputs, y_true = inputs.to(dtype=torch.float32, device=device), y_true.to(device=device)
                y_true = y_true + 128
            elif self.runner_type == self.RUNNER_TYPES['REG']:
                inputs, y_true = inputs.to(dtype=torch.float32, device=device), y_true.unsqueeze(1).to(dtype=torch.float32, device=device)
            else:
                raise Exception('Runner Error: runner_type undefined')

            # forward pass
            y_pred = self.model(inputs)
            # calculate loss
            loss = self.criterion(y_pred, y_true)

            # post-processing
            test_losses.append(loss.item())
            test_y_true += y_true.tolist()
            test_y_pred += y_pred.tolist()

        return test_losses, test_y_true, test_y_pred

    def inference(self, dataloader):
        self.model.eval()
        y_preds = []
        # iterate over all dataloader, one batch at a time
        for i, (inputs, y_true) in enumerate(dataloader):
            # pre-processing
            inputs = inputs.to(dtype=torch.float32, device=device)
            # forward pass (inference)
            y_pred = self.model(inputs)
            # append to list
            y_preds += y_pred.tolist()
        return np.array(y_preds)
