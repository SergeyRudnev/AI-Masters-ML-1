import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    raise NotImplementedError()


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):  # X - объекты тестовой выборки
        # Сначала считаем расстояния от тестовой выборки до обучающей
        Dist = self._metric_func(X, self._X)
        # обрабатываем случай, когда число соседей = n_neighbors, сортируем и выводим всё
        if self.n_neighbors == Dist.shape[1]:
            indices = Dist.argsort(axis=1)
            distances = np.take_along_axis(Dist, indices, axis=1)
            if return_distance:
                return distances, indices
            else:
                return indices
        else:
            # находим индексы n_neighbors лучших в каждой строке, то есть ближайших соседей
            indices = np.argpartition(Dist, self.n_neighbors, axis=1)[:, :self.n_neighbors]
            ranks_top = np.take_along_axis(Dist, indices, axis=1)  # выбираем этих лучших
            # np.argpartition возвращает неотсортриованный список лучших, поэтому сортируем его
            indices_top = ranks_top.argsort(axis=1)
            # получаем индексы отсортрованных ближайших соседей
            indices = np.take_along_axis(indices, indices_top, axis=1)
            distances = np.take_along_axis(Dist, indices, axis=1)  # получаем расстояния до ближайших соседей
            if return_distance:
                return distances, indices
            else:
                return indices
