"""Transformers for sklearn pipeline"""

import json

from sklearn.base import BaseEstimator, TransformerMixin

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
            return [en_preprocessor.process(x) for x in X]
        else:
            return [cn_preprocessor.process(x) for x in X]
