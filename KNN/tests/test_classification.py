from unittest import TestCase, mock

from itertools import combinations

import numpy as np

from knn.classification import KNNClassifier


def get_distances_indices(n_indices, n_neighbors):
    seed = np.random.RandomState(2789)

    indices = np.asarray(list(combinations(range(n_indices), n_neighbors)))
    distances = np.ones((len(indices), n_neighbors))
    distances = distances / (n_neighbors - np.arange(n_neighbors))
    distances = distances + seed.random(distances.shape)

    return distances, indices


class KNNClassifierTest(TestCase):
    def test_base_scenario_identity_euclidean(self):
        seed = np.random.RandomState(2789)
        X = seed.permutation(500).reshape(10, -1)
        y = np.arange(len(X))

        clf = KNNClassifier(n_neighbors=1, algorithm='my_own', metric='euclidean', weights='uniform')
        clf.fit(X, y)

        self.assertTrue(np.all(clf.predict(X) == y))

    def test_base_scenario_euclidean_full(self):
        seed = np.random.RandomState(2789)
        X = seed.permutation(500).reshape(10, -1)
        X_train, X_test = X[:4], X[6:]
        y_train = [0, 0, 1, 1]

        clf = KNNClassifier(n_neighbors=3, algorithm='my_own', metric='euclidean', weights='uniform')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_true = np.asarray([1, 0, 0, 1])

        self.assertTrue(np.all(y_true == y_pred))

    def test_base_scenario_identity_cosine(self):
        seed = np.random.RandomState(2789)

        X = np.zeros(shape=(10, 50))
        rows = seed.permutation(10)
        cols = seed.choice(50, size=len(rows), replace=False)
        X[rows, cols] = 1

        y = np.arange(len(X))

        clf = KNNClassifier(n_neighbors=1, algorithm='my_own', metric='cosine', weights='uniform')
        clf.fit(X, y)

        self.assertTrue(np.all(clf.predict(X) == y))

    def test_base_scenario_cosine_full(self):
        seed = np.random.RandomState(7792)

        X = np.zeros(shape=(10, 7))
        rows = np.repeat(seed.permutation(10), 3)
        cols = seed.choice(X.shape[1], size=len(rows), replace=True)
        X[rows, cols] = 1
        X = X * (1 + np.arange(X.shape[1]))[np.newaxis]

        X_train, X_test = X[:4], X[6:]
        y_train = [1, 0, 0, 1]

        clf = KNNClassifier(n_neighbors=3, algorithm='my_own', metric='cosine', weights='uniform')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_true = np.asarray([0, 0, 1, 0])

        self.assertTrue(np.all(y_true == y_pred))

    @mock.patch('knn.NearestNeighborsFinder.kneighbors',
                return_value=get_distances_indices(n_indices=7, n_neighbors=5))
    def test_base_scenario_no_weighted(self, kneigbors):
        clf = KNNClassifier(n_neighbors=5, algorithm='my_own', weights='uniform')
        clf.fit(None, [0, 1, 0, 1, 0, 1, 0, 1, 1])

        y_true = np.asarray([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
        y_pred = clf.predict(X=np.zeros_like(y_true))

        self.assertTrue(np.all(y_true == y_pred))

    @mock.patch('knn.NearestNeighborsFinder.kneighbors',
                return_value=get_distances_indices(n_indices=7, n_neighbors=5))
    def test_base_scenario_weighted(self, kneigbors):
        clf = KNNClassifier(n_neighbors=5, algorithm='my_own', weights='distance')
        clf.fit(None, [0, 1, 0, 1, 0, 1, 0, 1, 1])

        y_true = np.asarray([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = clf.predict(X=np.zeros_like(y_true))

        self.assertTrue(np.all(y_true == y_pred))
