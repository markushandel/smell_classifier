import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.base_model import ModelTrainer


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # Define the activation function after the second fully connected layer
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid for binary classification
        return x


class NeuralNetworkModel(ModelTrainer):
    def __init__(self, param_grid=None):
        super().__init__(param_grid)
        self.param_grid = param_grid if param_grid else {'epochs': 1000, 'lr': [0.001]}

    @staticmethod
    def build_model(input_size, hidden_size, lr, epochs, class_weights):
        model = SimpleNN(input_size, hidden_size)
        class_weights = torch.FloatTensor([0.5])  # assuming a binary classification problem
        criterion = nn.BCELoss(weight=class_weights.view(1, -1))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return PyTorchWrapper(model, criterion, optimizer, epochs=epochs)

    def train(self, x, y):
        best_score = 0
        best_params = {}
        input_size = x.shape[1]
        class_counts = np.bincount(y)
        class_weights = torch.FloatTensor(1.0 / class_counts)
        epochs = self.param_grid['epochs']

        for epoch in range(epochs):
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}]')
            for lr in self.param_grid['lr']:
                wrapper = self.build_model(input_size, 3, lr, epoch, class_weights=class_weights)
                wrapper.fit(x, y)
                y_pred = wrapper.predict(x)
                accuracy = accuracy_score(y, y_pred)
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {'epochs': epoch, 'lr': lr}
                    self.model = wrapper

        print(f"Best Score: {best_score}")
        print(f"Best Params: {best_params}")


class PyTorchWrapper:
    def __init__(self, model, criterion, optimizer, epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def fit(self, x, y):
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        for epoch in range(self.epochs):
            outputs = self.model(x_tensor)
            loss = self.criterion(outputs, y_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            outputs = self.model(x_tensor)
            predicted = (outputs > 0.5).float()  # Thresholding at 0.5 for binary classification
        return predicted.cpu().numpy().flatten()
