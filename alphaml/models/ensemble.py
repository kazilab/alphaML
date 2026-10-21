import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SoftVotingEnsemble(ClassifierMixin, BaseEstimator):
    """
    Binary classifier averaging the positive-class probabilities of several alphaML models.

    members: [(model name, params), ...] with names from models.specs.
    threads: thread count for members that support one (None: library default).
    """

    def __init__(self, members=(), threads=None):
        self.members = members
        self.threads = threads

    def fit(self, X, y):
        from .folds import fit
        from .specs import MODELS

        self.estimators_ = [fit(MODELS[name], params, X, y, final=True, threads=self.threads)
                            for name, params in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([estimator.predict_proba(X)[:, 1] for estimator in self.estimators_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
