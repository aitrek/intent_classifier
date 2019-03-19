"""NLP preprocessing functions"""

from typing import List
from abc import abstractmethod


class Preprocessor:

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Preprocess text

        Parameters
        ----------
        text

        Returns
        -------
        Preprocessed text.

        """
        raise NotImplementedError


class EnPreprocessor(Preprocessor):

    def process(self, text: str):
        # todo
        return text


class CnPreprocessor(Preprocessor):

    def __init__(self):
        import jieba
        self._seg = jieba

    def add_words(self, user_dict: list=None):
        """
        Add user-defined words.

        Parameters
        ----------
        user_dict: user dict in 3 columns, like:
            [
                [word, word_frequency, pos_tag],
                [word, word_frequency, None],
                [word, None, None],
                ...
            ]
            The word frequency and POS tag can be omitted respectively.
            The word frequency will be filled with a suitable value if omitted.

        """
        for word, freq, pos in user_dict:
            if not word:
                continue
            self._seg.add_word(word, freq, pos)

    def del_words(self, del_words: List[str]):
        """
        Delete words.

        Parameters
        ----------
        del_words: The list of words to deleted.

        """
        for word in del_words:
            self._seg.del_word(word)

    def suggest_freq(self, segments: list):
        """
        Tune the frequency of words.

        Parameters
        ----------
        segments: The segments that the word is expected to be cut into,
                  If the word should be treated as a whole, use a str.


        Returns
        -------

        """
        for segment in segments:
            self._seg.suggest_freq(segment, tune=True)

    def process(self, text: str) -> str:
        return " ".join(self._seg.cut(text))


en_preprocessor = EnPreprocessor()
cn_preprocessor = CnPreprocessor()
