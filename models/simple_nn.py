import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from libs import load_data, preprocess_data, split_data


def create_data():
    X = torch.rand((100, 61))
    Y = torch.randint(0, 2, (100,))
    return X, Y


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
        x = self.softmax(self.fc4(x))  # Apply softmax to the final layer
        return x


def train_model(model, criterion, optimizer, X, Y, epochs):
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


def save_model(model, path='simple_nn_model.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model Saved at {path}')


def load_model(path='simple_nn_model.pth'):
    model = SimpleNN(input_size=61, hidden_size=100, output_size=2)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Model Loaded from {path}')
    return model


def load_arff(file_path):
    df = load_data(f'{file_path}')
    x, y = preprocess_data(df)
    return x,y


# Test Model
def test_model(model, X, Y):
    X_tensor = torch.FloatTensor(X)  # Directly create a FloatTensor
    Y_tensor = torch.LongTensor(Y)   # Directly create a LongTensor
    output = model(X_tensor)
    _, predicted = torch.max(output, 1)

    # Converting tensors to numpy arrays for sklearn metric calculation
    y_true = Y_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    accuracy = (predicted == Y_tensor).sum().item() / len(Y_tensor)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Model Accuracy: {accuracy}")
    print(f"Macro averaged F1 score: {f1}")


def main():
    # Creating the dataset
    x, y = load_arff("data/data-class.arff")
    x_train, x_test, y_train, y_test = split_data(x, y)

    x_train, x_test = torch.FloatTensor(x_train), torch.FloatTensor(x_test)
    y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

    # Creating an instance of the model
    model = SimpleNN(input_size=61, hidden_size=4, output_size=2)

    # Defining the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 10000
    trained_model = train_model(model, criterion, optimizer, x_train, y_train, epochs)

    test_model(trained_model, x_test, y_test)
    # Save the trained model
    save_model_path = 'simple_nn_model.pth'
    save_model(trained_model, save_model_path)


if __name__ == '__main__':
    main()
