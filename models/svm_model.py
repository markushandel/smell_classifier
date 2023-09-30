from sklearn.svm import SVC
from models.base_model import ModelTrainer

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
