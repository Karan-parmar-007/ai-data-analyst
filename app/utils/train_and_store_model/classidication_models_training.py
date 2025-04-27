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
                metric=hyperparameters.get('metric', 'minkowski')
            )
        elif model_name == "LogisticRegression":
            return logistic_regression(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                penalty=hyperparameters.get('penalty', 'l2'),
                C=hyperparameters.get('C', 1.0),
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
                max_depth=hyperparameters.get('max_depth', None),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1)
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
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1)
            )
        elif model_name == "GradientBoostingClassifier":
            return gradient_boosting_classifier(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_estimators=hyperparameters.get('n_estimators', 100),
                learning_rate=hyperparameters.get('learning_rate', 0.1),
                max_depth=hyperparameters.get('max_depth', 3),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1)
            )
        elif model_name == "AdaBoostClassifier":
            return adaboost_classifier(
                self.dataset_id,
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                n_estimators=hyperparameters.get('n_estimators', 50),
                learning_rate=hyperparameters.get('learning_rate', 1.0),
                algorithm=hyperparameters.get('algorithm', 'SAMME.R')
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
                gamma=hyperparameters.get('gamma', 'scale')
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
