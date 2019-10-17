import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from plsda import PLSDADummy


def create_classifiers(n_features):
    classifiers = [
        {
            'name': 'KNN1',
            'model': KNeighborsClassifier(1, n_jobs=4)
        },
        {
            'name': 'PLSDA-40',
            'model': PLSDADummy(40)
        },
        {
            'name': 'SVM',
            'model': SVC(kernel="linear", C=0.025, cache_size=7000, tol=0.01)
        },
        {
            'name': 'RF',
            'model': RandomForestClassifier(
                n_estimators=100, max_features=int(np.sqrt(n_features)) + 1,
                max_depth=None, min_samples_split=2, n_jobs=4)
        },
    ]
    return classifiers
