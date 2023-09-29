import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from libs import split_data
from models.base_model import ModelTrainer
import copy  # Import the copy module


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class NeuralNetworkModel(ModelTrainer):
    def __init__(self, param_grid=None):
        super().__init__(param_grid)
        self.param_grid = param_grid if param_grid else {'epochs': 5000, 'lr': [0.01, 0.001, 0.0001]}

    @staticmethod
    def build_model(input_size, hidden_size, lr, epochs, class_weights):
        model = SimpleNN(input_size, hidden_size)
        # Assuming class_weights is a list like [weight_for_class_0, weight_for_class_1]
        pos_weight = torch.FloatTensor([class_weights])  # using the weight for the positive class

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = nn.BCELoss(weight=class_weights.view(1, -1))
        optimizer = optim.Adam(model.parameters(), lr=lr)

        return PyTorchWrapper(model, criterion, optimizer)

    def train(self, x, y):
        best_score = 0
        best_params = {}
        input_size = x.shape[1]
        class_counts = np.bincount(y)
        total = np.sum(class_counts)
        class_weights = torch.FloatTensor([total / class_counts[1]])  # For binary classification pos_weight
        x_train, x_val, y_train, y_val = split_data(x, y, test_size=0.2)  # Adjust test_size as needed.

        for lr in self.param_grid['lr']:
            wrapper = self.build_model(input_size, 3, lr, self.param_grid['epochs'], class_weights=class_weights)
            for epoch in range(self.param_grid['epochs']):
                wrapper.fit(x_train, y_train)

                # Evaluate on validation set
                y_val_pred = wrapper.predict(x_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)

                if (epoch + 1) % 10 == 0 or epoch == self.param_grid['epochs'] - 1:
                    print(f'Epoch [{epoch + 1}/{self.param_grid["epochs"]}], LR: {lr}, Accuracy: {val_accuracy}')

                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = {'epochs': epoch, 'lr': lr}
                    self.model = copy.deepcopy(wrapper)  # Consider using deep copy to store the best model

        print(f"Best Score: {best_score}")
        print(f"Best Params: {best_params}")


class PyTorchWrapper:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, x, y):
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
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
