import numpy as np


def euclidean_distance(x, y):
    u = np.diagonal((x @ x.T)).reshape(x.shape[0], 1)   # квадраты векторов из матрицы x
    v = np.diagonal(y @ y.T).reshape(1, y.shape[0])     # квадраты векторов из матрицы y
    # приводим к нужной размерности, так как на выходе должна быть матрица попарных расстояний
    u = u @ np.ones(y.shape[0]).reshape(1, y.shape[0])
    v = np.ones(x.shape[0]).reshape(x.shape[0], 1) @ v
    return np.sqrt(u + v - 2*(x @ y.T))  # вычитаем попарные скалярные произведения векторов из x и векторов из y


def cosine_distance(x, y):
    u = np.sqrt(np.diagonal((x @ x.T)).reshape(x.shape[0], 1))  # длины векторов из матрицы x
    v = np.sqrt(np.diagonal(y @ y.T).reshape(1, y.shape[0]))    # длины векторов из матрицы y
    # приводим к нужной размерности, так как на выходе должна быть матрица попарных расстояний
    u = u @ np.ones(y.shape[0]).reshape(1, y.shape[0])
    v = np.ones(x.shape[0]).reshape(x.shape[0], 1) @ v
    return 1 - x @ y.T / u / v          # делим скалярные произведения векторов их на длины
