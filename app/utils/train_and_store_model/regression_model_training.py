from app.models_for_training.regression_models.linear_regression import linear_regression
from app.models_for_training.regression_models.lasso import lasso
from app.models_for_training.regression_models.ridge import ridge
from app.models_for_training.regression_models.elastic_net import elastic_net
from app.models_for_training.regression_models.ada_boost_regressor import adaboost_regressor
from app.models_for_training.regression_models.support_vector_regression import svr
import numpy as np

class RegressionModelTraining:
    def __init__(self, dataset_id: str, X_train, X_test, y_train, y_test):
        self.dataset_id = dataset_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model_name: str, hyperparameters: dict):
        if model_name == "linear_regression":
            # Validate hyperparameters
            if not isinstance(hyperparameters, dict):
                logging.error(f"Invalid hyperparameters for model {model_name}: {hyperparameters}. Expected a dictionary.")
                raise ValueError(f"Hyperparameters must be a dictionary, got {type(hyperparameters)}: {hyperparameters}")

            # Validate and convert n_jobs (from previous fix)
            n_jobs = hyperparameters.get('n_jobs', None)
            if isinstance(n_jobs, list):
                if 'None' in n_jobs:
                    n_jobs = None
                elif any(isinstance(x, int) for x in n_jobs):
                    n_jobs = next(x for x in n_jobs if isinstance(x, int))
                else:
                    n_jobs = None
            elif not (n_jobs is None or isinstance(n_jobs, int)):
                n_jobs = None

            result = linear_regression(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                fit_intercept=np.bool_(hyperparameters.get('fit_intercept', True)),
                copy_X=np.bool_(hyperparameters.get('copy_X', True)),
                n_jobs=n_jobs,
                positive=np.bool_(hyperparameters.get('positive', False))
            )
            if result and isinstance(result, dict):
                result["artifact_store_response"] = {"model_file_id": result["artifact_store_response"]}
            return result
        
        elif model_name == "ridge":
            return ridge(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                fit_intercept=np.bool_(hyperparameters.get('fit_intercept', True)),
                copy_X=np.bool_(hyperparameters.get('copy_X', True)),
                max_iter=hyperparameters.get('max_iter', None),
                tol=hyperparameters.get('tol', 0.0001),
                solver=hyperparameters.get('solver', 'auto'),
                positive=np.bool_(hyperparameters.get('positive', False)),
                random_state=hyperparameters.get('random_state', None)
            )

        elif model_name == "lasso":
            return lasso(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                fit_intercept=np.bool_(hyperparameters.get('fit_intercept', True)),
                precompute=np.bool_(hyperparameters.get('precompute', False)),
                warm_start=np.bool_(hyperparameters.get('warm_start', False)),
                positive=np.bool_(hyperparameters.get('positive', False)),
                random_state=hyperparameters.get('random_state', None),
                selection=hyperparameters.get('selection', 'cyclic'),
                copy_X=np.bool_(hyperparameters.get('copy_X', True)),
                max_iter=hyperparameters.get('max_iter', None),
                tol=hyperparameters.get('tol', 1e-3),
            )
        elif model_name == "elastic_net":
            return elastic_net(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                alpha=hyperparameters.get('alpha', 1.0),
                l1_ratio=hyperparameters.get('l1_ratio', 0.5),
                fit_intercept=np.bool_(hyperparameters.get('fit_intercept', True)),
                precompute=np.bool_(hyperparameters.get('precompute', False)),
                copy_X=np.bool_(hyperparameters.get('copy_X', True)),
                warm_start=np.bool_(hyperparameters.get('warm_start', False)),
                positive=np.bool_(hyperparameters.get('positive', False)),
                random_state=hyperparameters.get('random_state', None),
                normalize=np.bool_(hyperparameters.get('normalize', False)),
                max_iter=hyperparameters.get('max_iter', 1000),
                tol=hyperparameters.get('tol', 1e-4),
                selection=hyperparameters.get('selection', 'cyclic')
            )
        elif model_name == "adaboost_regressor":
            return adaboost_regressor(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                estimator=hyperparameters.get('estimator', None),
                n_estimators=hyperparameters.get('n_estimators', 50),
                learning_rate=hyperparameters.get('learning_rate', 1.0),
                loss=hyperparameters.get('loss', 'linear'),
                random_state=hyperparameters.get('random_state', None)
            )
        elif model_name == "svr":
            return svr(
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
                epsilon=hyperparameters.get('epsilon', 0.1),
                shrinking=np.bool_(hyperparameters.get('shrinking', True)),
                cache_size=hyperparameters.get('cache_size', 200),
                verbose=np.bool_(hyperparameters.get('verbose', False)),
                max_iter=hyperparameters.get('max_iter', -1)
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
