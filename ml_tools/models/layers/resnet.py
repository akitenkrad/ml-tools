import torch
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.batch_norm_1 = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_layers)])
        self.linear_1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_layers)])
        self.batch_norm_2 = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.linear_2 = nn.ModuleList([nn.Linear(hidden_dim, input_dim) for _ in range(n_layers)])

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in range(self.n_layers):
            x = self.batch_norm_1[layer](x)
            x = self.activation(x)
            x = self.linear_1[layer](x)
            x = self.batch_norm_2[layer](x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_2[layer](x)
        return x


class ResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.batch_norm_1 = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_layers)])
        self.linear_1 = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(n_layers)])
        self.batch_norm_2 = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.linear_2 = nn.ModuleList([nn.Linear(hidden_dim, input_dim) for _ in range(n_layers)])

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_org = torch.clone(x)
        for layer in range(self.n_layers):
            x = self.batch_norm_1[layer](x)
            x = self.activation(x)
            x = self.linear_1[layer](x)
            x = self.batch_norm_2[layer](x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_2[layer](x)
            x = x + x_org
        return x


class Highway(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, activation=nn.LeakyReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.batch_norm_1 = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.2)
        self.activation = activation

        for layer in range(n_layers):
            gate = self.gate[layer]
            nn.init.constant_(gate.bias, -1.0)

    def forward(self, x):
        for layer in range(self.n_layers):
            x = self.batch_norm_1[layer](x)

            gate = torch.sigmoid(self.gate[layer](x))
            non_linear = self.activation(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear

            x = self.dropout(x)
        return x
