import torch
import numpy as np
from torch.autograd import Variable

from libs import load_data, validate_data, preprocess_data
from models.simple_nn import SimpleNN


# Load Model
def load_model(path):
    model = SimpleNN(input_size=61, hidden_size=100, output_size=2)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Model Loaded from {path}')
    return model


# Load ARFF file
def load_arff(file_path):
    df = load_data(f'{file_path}')
    x, y = preprocess_data(df)
    return x,y


# Test Model
def test_model(model, X, Y):
    X_tensor = Variable(torch.Tensor(X))
    Y_tensor = torch.Tensor(Y).long()  # Convert Y to a Long Tensor
    output = model(X_tensor)
    _, predicted = torch.max(output, 1)
    accuracy = (predicted == Y_tensor).sum().item() / len(Y_tensor)  # Compare with Y_tensor
    print(f"Model Accuracy: {accuracy}")


# Main Function to Load ARFF, Model and Test
def main():
    file_path = 'data/data-class.arff'
    model_path = 'models/simple_nn_model.pth'

    X, Y = load_arff(file_path)
    model = load_model(model_path)
    test_model(model, X, Y)


if __name__ == '__main__':
    main()
