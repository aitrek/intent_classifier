"""Transformers for sklearn pipeline"""

import json

from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import Preprocessor


class Json2Dict(BaseEstimator, TransformerMixin):
    """Transform json string to Dict"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [json.loads(x) if x else {} for x in X]


class TextPreprocess(BaseEstimator, TransformerMixin):
    """Preprocess text"""

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocessor.process(x) for x in X]
