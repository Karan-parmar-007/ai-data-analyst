class GetModelsNamesAndHyperParameters:
    def __init__(self):
        pass
        

    def get_names(self, high_level_model):
        if high_level_model == 'classification_models':
            return ['ada boost classifier', 'decision tree classifier', 'gradient boost classifer', 'k nearest neighbors', 'logistic regression', 'random forest classifier', 'support vector classifier']
        elif high_level_model == 'regression_models':
            return ['ada boost regressor', 'elastic net', 'lasso', 'linear regression', 'ridge', 'support vector regressor']
        elif high_level_model == 'clustering':
            return ['k means clustering', 'agglomerative clustering', 'dbscan', 'mean shift', 'spectral clustering', 'mini batch k means', 'birch']
        elif high_level_model == 'naive_bayes':
            return ['Gaussian Naive Bayes', 'Multinomial Naive Bayes', 'Bernoulli Naive Bayes', 'Complement Naive Bayes', 'Categorical Naive Bayes']
        else:
            return []
