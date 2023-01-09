from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, algorithm, scoring, metric, weights, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    i = 0
    scores = {}
    k_max = np.max(k_list)
    models_KFold = list(np.zeros(cv.get_n_splits()))
    distances_KFold = list(np.zeros(cv.get_n_splits()))
    indices_KFold = list(np.zeros(cv.get_n_splits()))
    score_KFold = list(np.zeros(cv.get_n_splits()))

    for train_index, test_index in cv.split(X):  # разбиваем обучающую выборку на фолды
        for k in np.sort(k_list)[::-1]:
            if k == k_max:
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                models_KFold[i] = BatchedKNNClassifier(n_neighbors=k_max, algorithm=algorithm,
                                                       metric=metric, weights=weights)
                models_KFold[i].fit(x_train, y_train)
                # считаем расстояния и индексы соседей только для большего k
                distances_KFold[i], indices_KFold[i] = models_KFold[i].kneighbors(x_test, return_distance=True)
                y_pred = models_KFold[i]._predict_precomputed(indices_KFold[i], distances_KFold[i])
                scores[k] = accuracy_score(y_test, y_pred)  # считаем скоры для всех k
            else:
                # выбираем нужное количество сеседей
                y_pred = models_KFold[i]._predict_precomputed(indices_KFold[i][:, :k], distances_KFold[i][:, :k])
                scores[k] = accuracy_score(y_test, y_pred)
        score_KFold[i] = list(scores.values())
        i += 1
    score_KFold = np.asarray(score_KFold)
    scores = {}
    for k in range(len(k_list)):   # преобразуем вывод к нужному виду
        scores[str(np.sort(k_list)[::-1][k])] = score_KFold[:, k]
    return scores  # возвращаем значения accuracy для каждого k
