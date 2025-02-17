import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from models import PolynomialApproximator


def train_model(num_epochs, learning_rate, num_hidden_layers, hidden_dim, func_choice):
    if func_choice == "f(x)=x^3+2x":
        func = lambda x: x ** 3 + 2 * x
        func_name = "x^3 + 2x"
    else:
        func = lambda x: x ** 2
        func_name = "x^2"

    x_start, x_end, n_points = -2, 2, 200
    x_values = np.linspace(x_start, x_end, n_points)
    y_values = func(x_values)

    X = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(y_values, dtype=torch.float32).unsqueeze(1)

    model = PolynomialApproximator(input_dim=1,
                                   hidden_dim=hidden_dim,
                                   num_hidden_layers=num_hidden_layers,
                                   output_dim=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        predicted = model(X).numpy()

    return model, x_values, y_values, predicted, loss_history, func_name
