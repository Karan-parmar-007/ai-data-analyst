from app.models_for_training.naive_bayes.gaussian_naive_bayes import gaussian_nb
from app.models_for_training.naive_bayes.bernoulli_naive_bayes import bernoulli_nb
from app.models_for_training.naive_bayes.categorical_naive_bayes import categorical_nb
from app.models_for_training.naive_bayes.complement_naive_bayes import complement_nb
from app.models_for_training.naive_bayes.multinomial_naive_bayes import multinomial_nb

class NaiveBayesModelTraining:
    def __init__(self, dataset_id: str, X_train, X_test, y_train, y_test):
        self.dataset_id = dataset_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model_name: str, hyperparameters: dict):
        if model_name == "GaussianNB":
            return gaussian_nb(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                priors=hyperparameters.get('priors', None),
                var_smoothing=hyperparameters.get('var_smoothing', 1e-9)
            )
        elif model_name == "BernoulliNB":
            return bernoulli_nb(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                binarize=hyperparameters.get('binarize', 0.0),
                fit_prior=hyperparameters.get('fit_prior', True),
                class_prior=hyperparameters.get('class_prior', None)
            )
        elif model_name == "CategoricalNB":
            return categorical_nb(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                fit_prior=hyperparameters.get('fit_prior', True),
                class_prior=hyperparameters.get('class_prior', None),
                min_categories=hyperparameters.get('min_categories', None)
            )
        elif model_name == "ComplementNB":
            return complement_nb(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                fit_prior=hyperparameters.get('fit_prior', True),
                class_prior=hyperparameters.get('class_prior', None),
                norm=hyperparameters.get('norm', False)
            )
        elif model_name == "MultinomialNB":
            return multinomial_nb(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                fit_prior=hyperparameters.get('fit_prior', True),
                class_prior=hyperparameters.get('class_prior', None)
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
