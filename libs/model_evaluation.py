from sklearn.metrics import accuracy_score, f1_score


def print_evaluation(model_name, smell_name, data_type, accuracy, f1):
    print(f"The {data_type} accuracy is: ")
    print(f"Model: {model_name}, Smell: {smell_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("--------------------------")
