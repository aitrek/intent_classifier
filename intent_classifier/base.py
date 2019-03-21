"""Basic components"""
import numpy as np
from sklearn.utils import Bunch


class DataBunch(Bunch):
    """
    Subclass of sklearn Bunch which is used as container for datasets from
    kinds of data sources.

    It's attributes are confined to what is needed in the subsequent
    operations:
        words - User's words in the conversations
        contexts - Auxiliary information to get what user's intent.
        intents - User's intents
    """

    def __init__(self, words: np.array, contexts: np.array, intents: np.array):
        super().__init__(words=words, contexts=contexts, intents=intents)


class OneClassClassifier:
    """
    Classifier used for dataset which has only one class.
    """

    def __init__(self, intent: str):
        self._intent = intent

    def predict(self, X, **kwargs):
        return self._intent
