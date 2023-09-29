from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:
    def __init__(self, param_grid):
        self.model = None  # To be set by the subclass
        self.param_grid = param_grid

    def optimize_hyperparameters(self, x, y):
        grid_search = (
            GridSearchCV(self.model, self.param_grid, cv=10, scoring="accuracy", return_train_score=True, n_jobs=-1))
        grid_search.fit(x, y)
        self.model = grid_search.best_estimator_

    def train(self, x, y):
        self.optimize_hyperparameters(x, y)

    def evaluate_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred,
                      average='macro')  # 'macro' computes the F1-score for each label and then takes their average
        return accuracy, f1

