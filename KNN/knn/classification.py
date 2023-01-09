import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        # найдём метки соседей
        kneighbors_labels = np.take_along_axis(self._labels.reshape(-1, 1), indices.T, axis=0)
        if self._weights == 'distance':
            weights = 1 / (distances + self.EPS)  # веса обратно пропорциальны расстояниям до соседей
            # сделаем из меток сосдей 3d-array, взял пример преобразования со stackoverflow
            kneighbors_labels_3D = kneighbors_labels.T[:, :, np.newaxis]
            weighrs_3d = weights[:, :, np.newaxis]  # аналогично поступим с весами
            # размножим метки соседей по матрицам: по строкам соседи, по столбцам возможные метки,
            # а третья размерность - элементы тестовой выборки
            # сделали это, чтобы можно было по отдельности посчитать взвешенную сумму для каждого возможного лейбла
            mask = kneighbors_labels_3D == np.arange(self._labels.max() + 1)
            # заполним эту маску весами, для этого умножим её на веса
            weighted_mask = weighrs_3d * mask
            # посчитаем требуемую сумму по столбцам(3d-array, поэтому тут axis=1, то есть вдоль второй оси)
            res = np.sum(weighted_mask, axis=1)
            return np.argmax(res, axis=1)  # выберем индексы меток с максимальной суммой по строчкам
        else:
            def find_label(a):
                return np.argmax(np.bincount(a))
            return np.apply_along_axis(find_label, 0, kneighbors_labels)  # выберем самые часто встречаемые метки

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
            # super() - доступ к классу-наследнику, который не разбивает на батчи.
        else:
            import math as m
            number_of_steps = m.ceil(X.shape[0] / self._batch_size)
            if return_distance:
                # создаём два списка, в которые будем складывать индексы и расстояния для каждого куска
                batched_ind = list(np.zeros(number_of_steps, int))
                batched_dist = list(np.zeros(number_of_steps))
                for i in range(number_of_steps):  # выполняем поиск ближающих соседей для каждого батча
                    batched_dist[i], batched_ind[i] = super().kneighbors(
                        X[self._batch_size*i:self._batch_size*(i+1), :], return_distance=return_distance)
                batched_ind = np.vstack(batched_ind)  # собираем результаты вместе
                batched_dist = np.vstack(batched_dist)
                return batched_dist, batched_ind
            else:
                return np.vstack([super().kneighbors(X[self._batch_size*i:self._batch_size*(i+1), :],
                                                     return_distance=return_distance) for i in range(number_of_steps)])
