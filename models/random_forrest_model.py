from models.base_model import ModelTrainer
from sklearn.ensemble import RandomForestClassifier

default_param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [None, 10, 15, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [10, 20, 100],
}


class RandomForestModel(ModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = default_param_grid
        super().__init__(param_grid)
        self.model = RandomForestClassifier()
