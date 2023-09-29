from libs import load_data, preprocess_data, split_data, validate_data, print_evaluation
from models import NaiveBayesModel, RandomForestModel, DecisionTreeModel, SVMModel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for data in ["data-class", "feature-envy", "god-class", "long-method"]:
        df = load_data(f'data/{data}.arff')

        if not validate_data(df):
            continue

        x, y = preprocess_data(df)
        x_train, x_test, y_train, y_test = split_data(x, y)

        # Train and evaluate Decision Tree model
        for model in [SVMModel(), NaiveBayesModel(), RandomForestModel(), DecisionTreeModel()]:
            model.train(x_train, y_train)
            test_accuracy, test_f1 = model.evaluate_model(x_test, y_test)
            print_evaluation(model, data, "test", test_accuracy, test_f1)
