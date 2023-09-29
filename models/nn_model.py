import numpy as np

from models.base_model import ModelTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score


class PyTorchWrapper:
    def __init__(self, model, criterion, optimizer, epochs=1000):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def fit(self, x, y):
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.LongTensor(y)
        for epoch in range(self.epochs):
            outputs = self.model(x_tensor)
            loss = self.criterion(outputs, y_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            output = self.model(x_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy()


class NeuralNetworkModel(ModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'epochs': [1000],  # You can add more options
                'lr': [0.001]  # You can add more options
            }
        super().__init__(param_grid)
        # self.model = self.build_model(61, 4, 2)

    def build_model(self, input_size, hidden_size, output_size, lr=0.001, epochs=1000, class_weights=None):
        model = SimpleNN(input_size, hidden_size, output_size)
        criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return PyTorchWrapper(model, criterion, optimizer, epochs=epochs)

    def optimize_hyperparameters(self, x, y):
        best_score = 0
        best_params = {}
        for epochs in self.param_grid.get('epochs', [1000]):
            for lr in self.param_grid.get('lr', [0.001]):
                input_size = x.shape[1]  # x_train should be your training data
                class_counts = np.bincount(y)
                class_weights = torch.FloatTensor(1.0 / class_counts)
                wrapper = self.build_model(input_size, 4, 2, lr, epochs, class_weights=class_weights)
                wrapper.fit(x, y)
                y_pred = wrapper.predict(x)
                accuracy = accuracy_score(y, y_pred)
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {'epochs': epochs, 'lr': lr}
                    self.model = wrapper
        print(f"Best Score: {best_score}")
        print(f"Best Params: {best_params}")


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))  # Apply softmax to the final layer
        return x