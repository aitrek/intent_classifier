"""Basic components"""

import numpy as np

from abc import abstractmethod
from typing import List, Union

from sklearn.utils import Bunch


class DatasetBunch(Bunch):
    """
    Subclass of sklearn Bunch which is used as container for datasets from
    kinds of data sources.

    It's attributes are confined to what is needed in the subsequent
    operations:
        words - User's words in the conversations
        contexts - Auxiliary information to get what user's intent.
                   None or empty list if no need of contexts data.
        intents - User's intent labels
    """

    def __init__(self, words: np.array, contexts: np.array, intents: np.array):
        super().__init__(words=words, contexts=contexts, intents=intents)


class RuleBunch(Bunch):
    """Container of rules of words and contexts."""

    def __init__(self, words_rules: List[str], context_rules: List[str],
                 intent_labels: List[List[str]]):
        """

        Parameters
        ----------
        words_rules: string list of words rules of regex
        context_rules: json string list of contexts.
                       None or empty list if no need of context rules.
        intents: user's intent labels
        """
        super().__init__(words_rules=words_rules, context_rules=context_rules,
                         intent_labels=intent_labels)


class Classifier:

    @abstractmethod
    def predict(self, words: str="", context: Union[str, dict]=None) -> List[str]:
        raise NotImplementedError
