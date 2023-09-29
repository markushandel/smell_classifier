from models.base_model import ModelTrainer
from sklearn.ensemble import RandomForestClassifier

default_param_grid = {
    'n_estimators': [10, 50],  # fewer trees
    # 'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 15, 30, 40],  # smaller depth
    'min_samples_split': [2, 5, 10],  # higher minimum samples per split
    'min_samples_leaf': [10, 20, 100],  # higher minimum samples per leaf
}


class RandomForestModel(ModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = default_param_grid
        super().__init__(param_grid)
        self.model = RandomForestClassifier()
