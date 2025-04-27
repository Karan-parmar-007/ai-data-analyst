from app.models_for_training.classification_models.ada_boost_classifier import adaboost_classifier
from app.models_for_training.classification_models.decision_tree_classifier import decision_tree_classifier
from app.models_for_training.classification_models.random_forest_classifier import random_forest_classifier
from app.models_for_training.classification_models.gradient_boosting_classifier import gradient_boosting_classifier
from app.models_for_training.classification_models.k_nearest_neighbors_classifier import knn_classifier
from app.models_for_training.classification_models.logistic_regression import logistic_regression
from app.models_for_training.classification_models.support_vector_classifier import svc


class ClassificationModeTraining:
    def __init__(self, dataset_id: str, X_train, X_test, y_train, y_test):
        self.dataset_id = dataset_id
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, model_name: str, hyperparameters: dict):
        if model_name == "KNeighborsClassifier":
            return knn_classifier(
                self.dataset_id, 
                self.X_train, 
                self.X_test, 
                self.y_train, 
                self.y_test,
                n_neighbors=hyperparameters.get('n_neighbors', 5),
                weights=hyperparameters.get('weights', 'uniform'),
                algorithm=hyperparameters.get('algorithm', 'auto'),
                leaf_size=hyperparameters.get('leaf_size', 30),
                p=hyperparameters.get('p', 2),
                metric=hyperparameters.get('metric', 'minkowski'),
                metric_params=hyperparameters.get('metric_params', None),
                n_jobs=hyperparameters.get('n_jobs', None)
            )
        elif model_name == "LogisticRegression":
            return logistic_regression(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                penalty=hyperparameters.get('penalty', 'l2'),
                dual=hyperparameters.get('dual', False),
                tol=hyperparameters.get('tol', 0.0001),
                C=hyperparameters.get('C', 1.0),
                fit_intercept=hyperparameters.get('fit_intercept', True),
                intercept_scaling=hyperparameters.get('intercept_scaling', 1),
                class_weight=hyperparameters.get('class_weight', None),
                random_state=hyperparameters.get('random_state', None),
                multi_class=hyperparameters.get('multi_class', 'ovr'),
                verbose=hyperparameters.get('verbose', 0),
                solver=hyperparameters.get('solver', 'lbfgs'),
                max_iter=hyperparameters.get('max_iter', 100)
            )
        elif model_name == "DecisionTreeClassifier":
            return decision_tree_classifier(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                criterion=hyperparameters.get('criterion', 'gini'),
                splitter=hyperparameters.get('splitter', 'best'),
                max_depth=hyperparameters.get('max_depth', None),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
                min_weight_fraction_leaf=hyperparameters.get('min_weight_fraction_leaf', 0.0),
                max_features=hyperparameters.get('max_features', None),
                random_state=hyperparameters.get('random_state', None),
                max_leaf_nodes=hyperparameters.get('max_leaf_nodes', None),
                min_impurity_decrease=hyperparameters.get('min_impurity_decrease', 0.0),
                class_weight=hyperparameters.get('class_weight', None),
                ccp_alpha=hyperparameters.get('ccp_alpha', 0.0),
                monotonic_cst=hyperparameters.get('monotonic_cst', None)
            )
        elif model_name == "RandomForestClassifier":
            return random_forest_classifier(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_estimators=hyperparameters.get('n_estimators', 100),
                criterion=hyperparameters.get('criterion', 'gini'),
                max_depth=hyperparameters.get('max_depth', None),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
                min_weight_fraction_leaf=hyperparameters.get('min_weight_fraction_leaf', 0.0),
                max_features=hyperparameters.get('max_features', 'sqrt'),
                max_leaf_nodes=hyperparameters.get('max_leaf_nodes', None),
                min_impurity_decrease=hyperparameters.get('min_impurity_decrease', 0.0),
                bootstrap=hyperparameters.get('bootstrap', True),
                oob_score=hyperparameters.get('oob_score', False),
                n_jobs=hyperparameters.get('n_jobs', None),
                random_state=hyperparameters.get('random_state', None),
                verbose=hyperparameters.get('verbose', 0),
                warm_start=hyperparameters.get('warm_start', False),
                class_weight=hyperparameters.get('class_weight', None),
                ccp_alpha=hyperparameters.get('ccp_alpha', 0.0),
                max_samples=hyperparameters.get('max_samples', None),
                monotonic_cst=hyperparameters.get('monotonic_cst', None)
            )
        elif model_name == "GradientBoostingClassifier":
            return gradient_boosting_classifier(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                loss=hyperparameters.get('loss', 'log_loss'),
                criterion=hyperparameters.get('criterion', 'friedman_mse'),
                max_features=hyperparameters.get('max_features', None),
                min_impurity_decrease=hyperparameters.get('min_impurity_decrease', 0.0),
                min_weight_fraction_leaf=hyperparameters.get('min_weight_fraction_leaf', 0.0),
                subsample=hyperparameters.get('subsample', 1.0),
                n_estimators=hyperparameters.get('n_estimators', 100),
                learning_rate=hyperparameters.get('learning_rate', 0.1),
                max_depth=hyperparameters.get('max_depth', 3),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
                random_state=hyperparameters.get('random_state', None),
                verbose=hyperparameters.get('verbose', 0),
                warm_start=hyperparameters.get('warm_start', False),
                validation_fraction=hyperparameters.get('validation_fraction', 0.1),
                n_iter_no_change=hyperparameters.get('n_iter_no_change', None),
                tol=hyperparameters.get('tol', 0.0001),
                ccp_alpha=hyperparameters.get('ccp_alpha', 0.0),
                init=hyperparameters.get('init', None),
            )
        elif model_name == "AdaBoostClassifier":
            return adaboost_classifier(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                estimator=hyperparameters.get('estimator', None),
                n_estimators=hyperparameters.get('n_estimators', 50),
                learning_rate=hyperparameters.get('learning_rate', 1.0),
                algorithm=hyperparameters.get('algorithm', 'SAMME.R'),
                random_state=hyperparameters.get('random_state', None)
            )
        elif model_name == "SVC":
            return svc(
                self.dataset_id,
                X_train=self.X_train,
                X_test=self.X_test,
                y_train=self.y_train,
                y_test=self.y_test,
                C=hyperparameters.get('C', 1.0),
                kernel=hyperparameters.get('kernel', 'rbf'),
                degree=hyperparameters.get('degree', 3),
                gamma=hyperparameters.get('gamma', 'scale'),
                coef0=hyperparameters.get('coef0', 0.0),
                shrinking=hyperparameters.get('shrinking', True),
                probability=hyperparameters.get('probability', False),
                tol=hyperparameters.get('tol', 0.001),
                cache_size=hyperparameters.get('cache_size', 200),
                class_weight=hyperparameters.get('class_weight', None),
                verbose=hyperparameters.get('verbose', False),
                max_iter=hyperparameters.get('max_iter', -1),
                decision_function_shape=hyperparameters.get('decision_function_shape', 'ovr'),
                break_ties=hyperparameters.get('break_ties', False),
                random_state=hyperparameters.get('random_state', None)
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
