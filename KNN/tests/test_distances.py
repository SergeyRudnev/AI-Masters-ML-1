from unittest import TestCase

import numpy as np
from scipy.spatial.distance import cdist

from knn.distances import euclidean_distance, cosine_distance


class EuclidianDistanceTest(TestCase):
    def test_base_scenario_norm(self):
        x = np.eye(N=3, M=5) / np.sqrt(2)
        shape = (len(x), len(x), )

        xx_pred = euclidean_distance(x, x)
        xx_true = np.ones(shape) - np.eye(*shape)

        self.assertTrue(np.allclose(xx_pred, xx_true))

    def test_base_scenario_zeros(self):
        x = [[2, 5, 1, 1, 1, 1, 4, 0],
             [3, 5, 6, 3, 1, 1, 0, 0],
             [4, 5, 4, 7, 2, 3, 1, 1]]
        x = np.asarray(x)

        xx_pred = euclidean_distance(x, np.zeros_like(x)[:1])
        xx_true = np.asarray([7, 9, 11])[:, None]

        self.assertTrue(np.allclose(xx_pred, xx_true))

    def test_base_scenario_common(self):
        seed = np.random.RandomState(9872)

        x = seed.random(size=(20, 7))
        y = seed.random(size=(15, 7))

        xx_pred = euclidean_distance(x, y)
        xx_true = cdist(x, y)

        self.assertTrue(np.allclose(xx_pred, xx_true))


class CosineDistanceTest(TestCase):
    def test_base_scenario_norm(self):
        x = np.eye(N=3, M=5) / np.sqrt(2)
        shape = (len(x), len(x), )

        xx_pred = cosine_distance(x, x)
        xx_true = np.ones(shape) - np.eye(*shape)

        self.assertTrue(np.allclose(xx_pred, xx_true))

    def test_base_scenario_common(self):
        seed = np.random.RandomState(9872)

        x = seed.random(size=(20, 7))
        y = seed.random(size=(15, 7))

        xx_pred = cosine_distance(x, y)
        xx_true = cdist(x, y, metric='cosine')

        self.assertTrue(np.allclose(xx_pred, xx_true))
