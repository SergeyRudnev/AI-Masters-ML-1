import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = False

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        return np.mean(np.logaddexp(0, -(y * X.dot(w)))) + self.l2_coef * w[1:] @ w[1:]
        # np.logaddexp(x1,x2) = np.log(np.exp(x1) + np.exp(x2))

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        # создаём диагональную csr_matrix
        indptr = np.arange(X.shape[0]+1)  # суммарное количество ненулевых элементов, после каждой строчки
        indices = np.arange(X.shape[0])  # индексы ненулевыйх элементов в каждой строчке
        data = -y*expit(-(y * X.dot(w)))  # сами данные
        gradQ = np.asarray(np.mean(csr_matrix((data, indices, indptr),
                                              shape=(X.shape[0], X.shape[0])).dot(X), axis=0)).ravel()
        gradQ[1:] = gradQ[1:] + 2 * self.l2_coef * w[1:]
        return gradQ


class MultinomialLoss(BaseLoss):
    """
    Loss function for multinomial regression.
    It should support l2 regularization.
    w should be 2d numpy.ndarray.
    First dimension is class amount.
    Second dimesion is feature space dimension.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = True

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : float
        """
        pass

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : 2d numpy.ndarray
        """
        pass
