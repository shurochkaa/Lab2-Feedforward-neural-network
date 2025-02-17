import torch.nn as nn


class PolynomialApproximator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10, num_hidden_layers=1, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
