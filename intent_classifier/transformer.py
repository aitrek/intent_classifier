"""Transformers for sklearn pipeline"""

import json

from sklearn.base import BaseEstimator, TransformerMixin


class Json2Dict(BaseEstimator, TransformerMixin):
    """Transform json string to Dict"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [json.loads(x) if x else {} for x in X]
