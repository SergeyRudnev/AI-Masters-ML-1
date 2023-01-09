import numpy as np
from scipy.special import expit
import time


class LinearModel:
    def __init__(
            self,
            loss_function,
            batch_size=None,
            step_alpha=1,
            step_beta=0,
            tolerance=1e-5,
            max_iter=1000,
            random_seed=153,
            w=None,
            **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.w = w

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        import timeit
        epochs_time = []
        epochs_loss = []
        epochs_loss_val = []
        history = {}  # для сохранения истории обучения
        np.random.seed(self.random_seed)
        if w_0 is None:  # задаём начальное значение весов, если оно не задано
            w_0 = np.zeros(X.shape[1])
        k = 0
        diff = 100  # произвольное число, заведомо большее tolerance
        w_k = w_0
        if self.batch_size is None:  # обычный градиентный спуск с убывающим темпом обучения
            while (k < self.max_iter) & (diff > self.tolerance):
                if trace:
                    start_time = timeit.default_timer()
                step_eta = self.step_alpha / ((k + 1) ** self.step_beta)
                w_k1 = w_k - step_eta * self.loss_function.grad(X, y, w_k)  # шаг обучения
                diff = w_k1 @ w_k1.T + w_k @ w_k.T - 2 * w_k1 @ w_k.T
                w_k = w_k1
                k += 1
                if trace:
                    epochs_time.append(timeit.default_timer() - start_time)
                    epochs_loss.append(self.loss_function.func(X, y, w_k))
                    if (X_val is not None) & (y_val is not None):
                        epochs_loss_val.append(self.loss_function.func(X_val, y_val, w_k))
        else:  # стохастический градиентный спуск, сначала все признаки перемешиваются, а затем последовательно
            # выбирается выборка размера batch_size
            import math as m
            number_of_steps = m.ceil(X.shape[0] / self.batch_size)
            index = np.arange(X.shape[0])
            while (k < self.max_iter) & (diff > self.tolerance):
                # под max_iter понимаем количество проходов по всей обуч. выборке
                if trace:
                    start_time = timeit.default_timer()
                np.random.shuffle(index)  # случайно поменяем строчки матрицы местами
                step_eta = self.step_alpha / ((k + 1) ** self.step_beta)
                w_epoch_k = w_k
                for i in range(number_of_steps):  # выполняем градиентный спуск для каждого батча
                    w_epoch_k1 = w_epoch_k - step_eta * self.loss_function.grad(
                        X[index][self.batch_size * i:self.batch_size * (i + 1), :],
                        y[index][self.batch_size * i:self.batch_size * (i + 1)], w_epoch_k)
                    w_epoch_k = w_epoch_k1
                w_k1 = w_epoch_k
                diff = w_k1 @ w_k1.T + w_k @ w_k.T - 2 * w_k1 @ w_k.T
                w_k = w_k1
                k += 1
                if trace:
                    epochs_time.append(timeit.default_timer() - start_time)
                    epochs_loss.append(self.loss_function.func(X, y, w_k))
                    if (X_val is not None) & (y_val is not None):
                        epochs_loss_val.append(self.loss_function.func(X_val, y_val, w_k))
        self.w = w_k
        if trace:
            history['time'] = epochs_time
            history['func'] = epochs_loss
            if (X_val is not None) & (y_val is not None):
                history['func_val'] = epochs_loss_val
            return history

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        return np.sign(X.dot(self.get_weights()) - threshold)
        # то есть выдаём 1, если скалярное произведение превышает значение threshold

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threhold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.get_weights())