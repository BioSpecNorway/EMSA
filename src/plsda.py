from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class PLSDADummy(BaseEstimator):
    """
    Wrapper of PLSRegression for classification.

    PLSRegression predicts one hot encoded vectors,
    then plsda outputs class with maximal score.
    """
    def __init__(self, n_components=2):
        self.pls = PLSRegression(n_components)
        self.classes = None

    def __one_hot_encode(self, Y):
        # encode labels to numbers
        Y = np.array([np.where(self.classes == y)[0][0] for y in Y])

        enc = OneHotEncoder(n_values=len(self.classes))
        return enc.fit_transform(Y.reshape(-1, 1)).toarray()

    def fit(self, X, Y):
        """

        :param X:
        :param Y: list of labels
        :return : 
        """
        self.classes = np.array(sorted(np.unique(Y)))

        Y = self.__one_hot_encode(Y)
        self.pls.fit(X, Y)

        return self

    def predict(self, X):
        y_pred = np.argmax(self.pls.predict(X), axis=1)
        return np.array([self.classes[cls] for cls in y_pred])
