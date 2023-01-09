import numpy as np

from scipy.optimize import minimize_scalar
from scipy.special import expit

from sklearn.tree import DecisionTreeRegressor
from .sampler import FeatureSampler


class Booster:
    def __init__(self, base_estimator, feature_sampler, object_sampler, n_estimators=10, lr=.5, **params):
        """
        n_estimators : int
            number of base estimators
        base_estimator : class
            class for base_estimator with fit(), predict() and predict_proba() methods
        feature_sampler : instance of FeatureSampler
        n_estimators : int
            number of base_estimators
        lr : float
            learning rate for estimators
        params : kwargs
            kwargs for base_estimator init
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.object_sampler = object_sampler
        self.feature_sampler = feature_sampler
        self.estimators = []
        self.indices = []
        self.weights = []
        self.lr = lr
        self.params = params

    def _fit_first_estimator(self, X, y):
        ind = self.object_sampler.sample_indices(X.shape[0])
        x, y1 = X[ind], y[ind]

        ind_features = self.feature_sampler.sample_indices(X.shape[1])
        x = x[:, ind_features]

        base_est = self.base_estimator(**self.params)
        self.estimators.append(base_est.fit(x, y1))
        self.indices.append(ind_features)
        self.weights.append(1)
        return self

    def _fit_base_estimator(self, X, y, predictions):
        raise NotImplementedError

    def fit(self, X, y):
        """
        Calculate final predictions:
            1) fit first estimator
            2) fit next estimator based on previous predictions
            3) update predictions
            4) got to step 2
        Don't forget, that each estimator has its own feature indices for prediction
        """

        self.estimators = []
        self.indices = []
        self.weights = []

        self._fit_first_estimator(X, y)  # строим первое дерево и запоминаем его предсказания
        ay = self.estimators[0].predict(X[:, self.indices[0]])

        for estimator_n in range(1, self.n_estimators):
            # строим новое дерево и добавляем его к старому с некоторым весом
            self._fit_base_estimator(X, y, ay)  # здесь рассчитываем веса для новой модели
            ay += self.weights[-1] * self.estimators[-1].predict(X)  # добавляем новую модель

        return self


    def predict(self, X):
        """
        Returns
        -------
        predictions : numpy ndarrays of shape (n_objects, n_classes)

        Calculate final predictions:
            1) calculate first estimator predictions
            2) calculate updates from next estimator
            3) update predictions
            4) got to step 2
        Don't forget, that each estimator has its own feature indices for prediction
        """
        if not (0 < len(self.estimators) == len(self.indices) == len(self.weights)):
            raise RuntimeError('Booster is not fitted', (len(self.estimators), len(self.indices)))

        self.predictions = self.estimators[0].predict(X)
        for estimator_n in range(1, self.n_estimators):
            self.predictions += self.weights[-1] * self.estimators[-1].predict(X)  # предказание так же усредняются

        return self.predictions


class GradientBoostingClassifier(Booster):
    def __init__(self, n_estimators=30, max_features_samples=0.8, lr=.5, max_depth=None, min_samples_leaf=1,
                 random_state=None, **params):
        base_estimator = DecisionTreeRegressor
        feature_sampler = FeatureSampler(max_samples=max_features_samples, random_state=random_state)

        super().__init__(
            base_estimator=base_estimator,
            feature_sampler=feature_sampler,
            n_estimators=n_estimators,
            lr=lr,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **params,
        )

    def _gradient(self, y_true, y_pred):
        """
        Calculate gradient for NLL
        """
        raise NotImplementedError('Put your code here')

    def _loss(self, y_true, y_pred):
        """
        Calculate NLL
        """
        raise NotImplementedError('Put your code here')

    def _fit_base_estimator(self, X, y, predictions):
        """
        Fits next estimator:
            1) calculate gradient
            2) select random indices of features for current estimator
            3) fit base_estimator (don't forget to remain only selected features)
            4) save base_estimator (self.estimators) and feature indices (self.indices)

        NOTE that self.base_estimator is class and you should init it with
        self.base_estimator(**self.params) before fitting
        """
        raise NotImplementedError('Put your code here')

    def predict_proba(self, X):
        return np.clip(super().predict(X), 0, 1)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
