import numpy as np
from models.base_model import ModelTrainer
from sklearn.tree import DecisionTreeClassifier


default_param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(10, 21), 'min_samples_leaf': [1, 5, 10, 20, 50, 100]}


class DecisionTreeModel(ModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = default_param_grid

        super().__init__(param_grid)
        self.model = DecisionTreeClassifier()