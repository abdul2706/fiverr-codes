"""
Original file is located at
https://colab.research.google.com/drive/1d9oBD1JE3hI3TeIYlYJIycrsxOL6Tljs
"""

from dataset import train_dataloader
import numpy as np

# deep learning libraries
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# this variable will help using gpu if it's available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device:', device)

"""# Define Models

## ChessNetRegression
"""

class ChessNetRegression(nn.Module):
    def __init__(self, num_hidden_layers=1, use_reduction=False, debug=False):
        super(ChessNetRegression, self).__init__()
        self.TAG = '[ChessNetRegression]'
        self.debug = debug

        in_neurons = 770
        out_neurons = 512
        self.r = int(0.9 * 512 / num_hidden_layers) if use_reduction else 0
        
        # define network layers
        self.num_hidden_layers = num_hidden_layers
        self.nn_layers = nn.ModuleList()
        self.nn_layers.append(nn.Linear(in_neurons, out_neurons, bias=True))
        for i in range(num_hidden_layers):
            in_neurons = out_neurons
            out_neurons = int(in_neurons - self.r)
            self.nn_layers.append(nn.Linear(in_neurons, out_neurons, bias=True))
        self.nn_layers.append(nn.Linear(out_neurons, 1, bias=True))
        # define activation layer
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.20)

        self.init_params()

    def init_params(net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                # init.normal_(m.weight, std=0.2)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.debug: print(self.TAG, '[x]', x.shape)
        for layer in self.nn_layers[:-1]:
            x = self.dropout(self.sigmoid(layer(x)))
            if self.debug: print(self.TAG, '[x]', x.shape)
        x = self.nn_layers[-1](x)
        if self.debug: print(self.TAG, '[x]', x.shape)
        return x

"""## ChessNetRegression2"""

class ChessNetRegression2(nn.Module):
    def __init__(self, architecture, debug=False):
        super(ChessNetRegression2, self).__init__()
        self.TAG = '[ChessNetRegression2]'
        self.debug = debug

        # define network layers
        self.nn_layers = nn.ModuleList()
        in_neurons = 770
        for layer_info in architecture:
            if layer_info[0] == 'L':
                out_neurons = layer_info[1]
                self.nn_layers.append(nn.Linear(in_neurons, out_neurons, bias=True))
                in_neurons = out_neurons
            elif layer_info[0] == 'D':
                self.nn_layers.append(nn.Dropout(layer_info[1]))
        self.nn_layers.append(nn.Linear(out_neurons, 1, bias=True))

        # define activation layer
        self.relu = nn.ReLU()

        self.init_params()

    def init_params(net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                # init.normal_(m.weight, std=0.2)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.debug: print(self.TAG, '[x]', x.shape)
        for layer in self.nn_layers[:-1]:
            if isinstance(layer, nn.Linear):
                x = self.relu(layer(x))
            if isinstance(layer, nn.Dropout):
                x = layer(x)
            if self.debug: print(self.TAG, '[x]', x.shape)
        x = self.nn_layers[-1](x)
        if self.debug: print(self.TAG, '[x]', x.shape)
        return x

"""## ChessNetRegCNN"""

class ChessNetRegCNN(nn.Module):
    def __init__(self, architecture, debug=False):
        super(ChessNetRegCNN, self).__init__()
        self.TAG = '[ChessNetRegCNN]'
        self.debug = debug

        # define network layers
        self.nn_layers = nn.ModuleList()
        in_neurons = None
        in_channels = 1
        count = 1
        for layer_info in architecture:
            if layer_info[0] == 'C':
                out_channels = layer_info[1]
                kernel = layer_info[2]
                stride = 2 if count % 3 == 0 else 1
                if self.debug: print(self.TAG, in_channels, out_channels, kernel, stride)
                self.nn_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride))
                in_channels = out_channels
                count += 1
            elif layer_info[0] == 'L':
                if in_neurons is None:
                    in_neurons = self.calculate_out_neurons()
                    if self.debug: print(self.TAG, '[in_neurons]', in_neurons)
                out_neurons = layer_info[1]
                self.nn_layers.append(nn.Linear(in_neurons, out_neurons, bias=True))
                in_neurons = out_neurons
            elif layer_info[0] == 'D':
                self.nn_layers.append(nn.Dropout(layer_info[1]))
        # out_neurons = self.calculate_out_neurons()
        self.nn_layers.append(nn.Linear(out_neurons, 1, bias=True))

        # define activation layer
        self.relu = nn.ReLU()

        self.init_params()

    def init_params(net):
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)

    def calculate_out_neurons(self):
        x = torch.rand((1, 1, 770))
        for layer in self.nn_layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x)
        return np.product(x.shape)

    def forward(self, x):
        if self.debug: print(self.TAG, '[x]', x.shape)
        is_first_linear = True
        for layer in self.nn_layers[:-1]:
            if isinstance(layer, nn.Conv1d):
                x = self.relu(layer(x))
                if self.debug: print(self.TAG, '[x]', x.shape)
            if isinstance(layer, nn.Linear):
                if is_first_linear:
                    is_first_linear = False
                    x = x.view(x.shape[0], -1)
                    if self.debug: print(self.TAG, '[flatten]', x.shape)
                x = self.relu(layer(x))
                if self.debug: print(self.TAG, '[x]', x.shape)
            if isinstance(layer, nn.Dropout):
                x = layer(x)
        x = self.nn_layers[-1](x)
        if self.debug: print(self.TAG, '[x]', x.shape)
        return x

"""## Test Model Forward Pass"""
def get_model_size(model):
    return sum([np.prod(params.shape) for params in model.parameters()])

if __name__ == '__main__':
    # model_reg = ChessNetRegression(num_hidden_layers=3, debug=True).to(device)
    # model_reg = ChessNetRegression2(num_hidden_layers=3, debug=True).to(device)
    model_reg = ChessNetRegCNN(architecture=[*[['C', 8, 9], ['D', 0.10]] * 3, 
                                            *[['C', 16, 9], ['D', 0.10]] * 3, 
                                            ['L', 40], ['D', 0.10], ['L', 20], ['D', 0.10], ['L', 10], ['D', 0.10]], debug=True).to(device)
    print(model_reg)
    print('[model_reg]', get_model_size(model_reg))
    x, y_true = next(iter(train_dataloader))
    x, y_true = x.unsqueeze(1).to(dtype=torch.float32, device=device), y_true.unsqueeze(1).to(device=device)
    print('[x]', x.shape)
    print('[y_true]', y_true.shape, '\n', y_true[:10].flatten())
    y_pred = model_reg(x)
    print('[y_pred]', y_pred.detach().shape, '\n', y_pred.detach()[:10].flatten())
