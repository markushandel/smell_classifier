from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:
    def __init__(self, param_grid):
        self.model = None  # To be set by the subclass
        self.param_grid = param_grid
        self.grid_search = None

    def optimize_hyperparameters(self, x, y):
        self.grid_search = (
            GridSearchCV(self.model, self.param_grid, cv=10, scoring="accuracy", return_train_score=True, n_jobs=-1))
        self.grid_search.fit(x, y)
        self.model = self.grid_search.best_estimator_

    def train(self, x, y):
        self.optimize_hyperparameters(x, y)

    def predict(self, x, y):
        y_pred = self.model.predict(x)
        return y_pred

    def get_cv_results(self):
        return self.grid_search.cv_results_

    def evaluate_model(self, x_test, y_test):
        y_pred = self.predict(x_test, y_test)
        return self.evaluate_predictions(y_pred, y_test)

    @staticmethod
    def evaluate_predictions(y_pred, y):
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred,
                      average='macro')  # 'macro' computes the F1-score for each label and then takes their average
        return accuracy, f1
