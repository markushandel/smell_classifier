from sklearn.svm import SVC
from models.base_model import ModelTrainer
from sklearn.model_selection import RandomizedSearchCV

default_param_grid = {
    'C': [0.1],  # Reduce the number of values
    'kernel': ['rbf'],  # Limit the number of kernels
    'class_weight': [None]
}


class SVMModel(ModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = default_param_grid
        super().__init__(param_grid)
        self.model = SVC()

    # def optimize_hyperparameters(self, x, y):
    #     random_search = RandomizedSearchCV(self.model, param_distributions=self.param_grid,
    #                                        n_iter=1, cv=5, scoring='accuracy', n_jobs=-1)
    #     random_search.fit(x, y)
    #     self.model = random_search.best_estimator_
