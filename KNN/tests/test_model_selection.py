from unittest import TestCase

import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from knn.model_selection import knn_cross_val_score


def complex_roots(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.exp(angles * 1j)


def knn_cross_val_score_sklearn(X, y, k_list, scoring, cv=None, **kwargs):
    scores = {}

    for k in k_list:
        score = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, **kwargs),
            X, y=y, cv=cv, scoring=scoring,
        )
        scores[k] = score

    return scores


class KnnCrossValScoreTest(TestCase):
    def test_leave_one_out_simple_star(self):
        seed = np.random.RandomState(228)
        x = complex_roots(6)
        x = np.vstack([np.real(x), np.imag(x)]).T
        x += seed.random(x.shape) * 0.2

        y = np.ones(len(x), dtype=int)
        y[:len(y) // 2] = 0

        cv = LeaveOneOut()

        scores_pred = knn_cross_val_score(x, y, k_list=range(1, len(x), 2), cv=cv, scoring='accuracy')
        scores_true = knn_cross_val_score_sklearn(
            x, y, k_list=range(1, len(x), 2), cv=cv, scoring='accuracy',
            metric='euclidean', weights='uniform', algorithm='brute',
        )

        scores_pred = {k: list(v) for k, v in scores_pred.items()}
        scores_true = {k: list(v) for k, v in scores_true.items()}

        self.assertDictEqual(scores_true, scores_pred)

    def test_leave_one_out_simple_power(self):
        x = [2 ** a for a in range(15)]
        x = np.vstack([x, x]).T
        y = [i % 3 for i in range(5) for _ in range(5)]

        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        cv = LeaveOneOut()

        scores_pred = knn_cross_val_score(x, y, k_list=range(1, 8, 2), cv=cv, scoring='accuracy')
        scores_true = knn_cross_val_score_sklearn(
            x, y, k_list=range(1, 8, 2), cv=cv, scoring='accuracy',
            metric='euclidean', weights='uniform', algorithm='brute',
        )

        scores_pred = {k: list(v) for k, v in scores_pred.items()}
        scores_true = {k: list(v) for k, v in scores_true.items()}

        self.assertDictEqual(scores_true, scores_pred)

    def test_base_scenario(self):
        seed = np.random.RandomState(9872)

        x = seed.random(size=(200, 10)) * 2 - 1
        y = seed.randint(0, 5, size=len(x))

        cv = KFold(n_splits=5, shuffle=True, random_state=226)
        scores_pred = knn_cross_val_score(x, y, k_list=[1, 3, 5, 7], cv=cv, scoring='accuracy')

        scores_true = knn_cross_val_score_sklearn(
            x, y, k_list=[1, 3, 5, 7], cv=cv, scoring='accuracy',
            metric='euclidean', weights='uniform', algorithm='brute',
        )

        scores_pred = {k: list(v) for k, v in scores_pred.items()}
        scores_true = {k: list(v) for k, v in scores_true.items()}

        self.assertDictEqual(scores_true, scores_pred)

    def test_base_scenario_params(self):
        seed = np.random.RandomState(9872)

        x = seed.random(size=(200, 10)) * 2 - 1
        y = seed.randint(0, 5, size=len(x))

        cv = KFold(n_splits=5, shuffle=True, random_state=226)

        scores_pred = knn_cross_val_score(
            x, y, k_list=[1, 3, 5, 7], cv=cv, scoring='accuracy',
            metric='euclidean', weights='distance')

        scores_true = knn_cross_val_score_sklearn(
            x, y, k_list=[1, 3, 5, 7], cv=cv, scoring='accuracy',
            metric='euclidean', weights='distance', algorithm='brute',
        )

        scores_pred = {k: list(v) for k, v in scores_pred.items()}
        scores_true = {k: list(v) for k, v in scores_true.items()}

        self.assertDictEqual(scores_true, scores_pred)

        scores_pred = knn_cross_val_score(
            x, y, k_list=[1, 3, 5, 7], cv=cv, scoring='accuracy',
            metric='cosine', weights='distance',
        )

        scores_true = knn_cross_val_score_sklearn(
            x, y, k_list=[1, 3, 5, 7], cv=cv, scoring='accuracy',
            metric='cosine', weights='distance', algorithm='brute',
        )

        scores_pred = {k: list(v) for k, v in scores_pred.items()}
        scores_true = {k: list(v) for k, v in scores_true.items()}

        self.assertDictEqual(scores_true, scores_pred)
