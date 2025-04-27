from app.models_for_training.regression_models.linear_regression import linear_regression
from app.models_for_training.regression_models.lasso import lasso
from app.models_for_training.regression_models.ridge import ridge
from app.models_for_training.regression_models.elastic_net import elastic_net
from app.models_for_training.regression_models.ada_boost_regressor import ada_boost_regressor
from app.models_for_training.regression_models.support_vector_regression import support_vector_regression

class RegressionModelTraining:
    def __init__(self, dataset_id: str, X_train, X_test, y_train, y_test):
        self.dataset_id = dataset_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model_name: str, hyperparameters: dict):
        if model_name == "LinearRegression":
            return linear_regression(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                fit_intercept=hyperparameters.get('fit_intercept', True),
                copy_X=hyperparameters.get('copy_X', True),
                n_jobs=hyperparameters.get('n_jobs', None)
            )
        elif model_name == "Lasso":
            return lasso(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                fit_intercept=hyperparameters.get('fit_intercept', True),
                normalize=hyperparameters.get('normalize', False),
                max_iter=hyperparameters.get('max_iter', 1000),
                tol=hyperparameters.get('tol', 1e-4),
                selection=hyperparameters.get('selection', 'cyclic')
            )
        elif model_name == "Ridge":
            return ridge(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                fit_intercept=hyperparameters.get('fit_intercept', True),
                normalize=hyperparameters.get('normalize', False),
                copy_X=hyperparameters.get('copy_X', True),
                max_iter=hyperparameters.get('max_iter', None),
                tol=hyperparameters.get('tol', 1e-3),
                solver=hyperparameters.get('solver', 'auto')
            )
        elif model_name == "ElasticNet":
            return elastic_net(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                l1_ratio=hyperparameters.get('l1_ratio', 0.5),
                fit_intercept=hyperparameters.get('fit_intercept', True),
                normalize=hyperparameters.get('normalize', False),
                max_iter=hyperparameters.get('max_iter', 1000),
                tol=hyperparameters.get('tol', 1e-4),
                selection=hyperparameters.get('selection', 'cyclic')
            )
        elif model_name == "AdaBoostRegressor":
            return ada_boost_regressor(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_estimators=hyperparameters.get('n_estimators', 50),
                learning_rate=hyperparameters.get('learning_rate', 1.0),
                loss=hyperparameters.get('loss', 'linear'),
                random_state=hyperparameters.get('random_state', None)
            )
        elif model_name == "SVR":
            return support_vector_regression(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                kernel=hyperparameters.get('kernel', 'rbf'),
                degree=hyperparameters.get('degree', 3),
                gamma=hyperparameters.get('gamma', 'scale'),
                coef0=hyperparameters.get('coef0', 0.0),
                tol=hyperparameters.get('tol', 1e-3),
                C=hyperparameters.get('C', 1.0),
                epsilon=hyperparameters.get('epsilon', 0.1)
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
