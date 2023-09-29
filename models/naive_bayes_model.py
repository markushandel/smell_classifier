from models.base_model import ModelTrainer
from sklearn.naive_bayes import GaussianNB, BernoulliNB


class NaiveBayesModel(ModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = {}
        super().__init__(param_grid)
        self.model = BernoulliNB()  # Naive Bayes usually doesn’t have hyperparameters to optimize, but it’s added for uniformity.
