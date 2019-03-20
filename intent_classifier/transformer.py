"""Transformers for sklearn pipeline"""

import json
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

from .preprocess import en_preprocessor, cn_preprocessor


class Json2Dict(BaseEstimator, TransformerMixin):
    """Transform json string to Dict"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [json.loads(x) if x else {} for x in X]


class TextPreprocess(BaseEstimator, TransformerMixin):
    """Preprocess text"""

    def __init__(self, lang: str="en"):
        self.lang = lang

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.lang == "cn":
            return [cn_preprocessor.process(x) for x in X]
        else:
            return [en_preprocessor.process(x) for x in X]


class PercentSVD(TruncatedSVD):

    def __init__(self, percent: float=1, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        super().__init__(algorithm=algorithm, n_iter=n_iter,
                         random_state=random_state, tol=tol)
        self.percent = percent

    def _calc_n_components(self, total_components: int):
        return math.ceil((total_components - 1) * self.percent)

    def fit_transform(self, X, y=None):
        self.n_components = self._calc_n_components(X.shape[1])
        return super().fit_transform(X, y)
