from models.base_model import ModelTrainer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB

default_param_grid = {'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

class NaiveBayesModel(ModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = default_param_grid
        super().__init__(param_grid)
        self.model = BernoulliNB()  # Naive Bayes usually doesn’t have hyperparameters to optimize, but it’s added for uniformity.
