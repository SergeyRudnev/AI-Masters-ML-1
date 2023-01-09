from unittest import TestCase

import numpy as np
from scipy.spatial.distance import cdist

from knn.nearest_neighbors import NearestNeighborsFinder


class NearestNeighborsFinderTest(TestCase):
    def test_base_scenario_identity_euclidean(self):
        seed = np.random.RandomState(9872)
        X = seed.permutation(500).reshape(10, -1)

        nn = NearestNeighborsFinder(n_neighbors=1, metric='euclidean')
        nn.fit(X)

        distances, indices = nn.kneighbors(X, return_distance=True)
        self.assertTrue(np.all(np.arange(len(X))[:, np.newaxis] == indices))
        self.assertTrue(np.all(np.zeros(len(X))[:, np.newaxis] == distances))

    def test_base_scenario_euclidean(self):
        seed = np.random.RandomState(9872)
        X = seed.permutation(500).reshape(10, -1)
        X_train, X_test = X[:4], X[6:]

        nn = NearestNeighborsFinder(n_neighbors=3, metric='euclidean')
        nn.fit(X_train)

        distances_pred, indices_pred = nn.kneighbors(X_test, return_distance=True)

        distances_true = cdist(X_test, X_train)
        indices_true = np.argsort(distances_true, axis=1)[:, :nn.n_neighbors]
        distances_true = np.take_along_axis(distances_true, indices_true, axis=1)

        self.assertTrue(np.allclose(distances_true, distances_pred))
        self.assertTrue(np.all(indices_true == indices_pred))

    def test_base_scenario_identity_cosine(self):
        seed = np.random.RandomState(9872)

        X = np.zeros(shape=(10, 50))
        rows = seed.permutation(len(X))
        cols = seed.choice(50, size=len(rows), replace=False)
        X[rows, cols] = 1

        nn = NearestNeighborsFinder(n_neighbors=1, metric='cosine')
        nn.fit(X)

        distances, indices = nn.kneighbors(X, return_distance=True)
        self.assertTrue(np.all(np.arange(len(X))[:, np.newaxis] == indices))
        self.assertTrue(np.all(np.zeros(len(X))[:, np.newaxis] == distances))

    def test_base_scenario_cosine(self):
        seed = np.random.RandomState(9872)

        X = np.zeros(shape=(10, 7))
        rows = np.repeat(seed.permutation(len(X)), 3)
        cols = seed.choice(X.shape[1], size=len(rows), replace=True)
        X[rows, cols] = 1
        X = X * (1 + np.arange(X.shape[1]))[np.newaxis]

        X_train, X_test = X[:4], X[6:]

        nn = NearestNeighborsFinder(n_neighbors=3, metric='cosine')
        nn.fit(X_train)

        distances_pred, indices_pred = nn.kneighbors(X_test, return_distance=True)

        distances_true = cdist(X_test, X_train, metric='cosine')
        indices_true = np.argsort(distances_true, axis=1)[:, :nn.n_neighbors]
        distances_true = np.take_along_axis(distances_true, indices_true, axis=1)

        self.assertTrue(np.allclose(distances_true, distances_pred))
        self.assertTrue(np.all(indices_true == indices_pred))

    def test_base_return_distance_flag(self):
        # euclidean case
        seed = np.random.RandomState(9872)
        X = seed.permutation(500).reshape(10, -1)
        X_train, X_test = X[:4], X[6:]

        nn = NearestNeighborsFinder(n_neighbors=1, metric='euclidean')
        nn.fit(X_train)

        distances_pred, indices_pred = nn.kneighbors(X_test, return_distance=True)
        self.assertTrue(np.all(nn.kneighbors(X_test, return_distance=False) == indices_pred))

        # cosine case
        X = np.ones(shape=(10, 50))
        X = np.tril(X, k=0)

        rows = seed.permutation(10)
        X = X[rows]

        X_train, X_test = X[:4], X[6:]

        nn = NearestNeighborsFinder(n_neighbors=1, metric='cosine')
        nn.fit(X_train)

        distances_pred, indices_pred = nn.kneighbors(X_test, return_distance=True)
        self.assertTrue(np.all(nn.kneighbors(X_test, return_distance=False) == indices_pred))
