"""Self-defined vectorizer"""

import numpy as np

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerWithEntity(TfidfVectorizer):
    """Subclass of TfidfVectorizer to support entity types"""

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, ner=None):
        """

        Parameters
        ----------
        ner: instance of named entity recognition.
            Its output, taking "Allen like cake." for example,
            should be a list in form:
            [
                {'value': 'Allen', 'type': 'person', 'start': 0, 'end': 5},
                {'value': 'cake', 'type': 'food', 'start': 11, 'end': 15}
            ]
        Other Parameters: see comments of TfidfVectorizer
        """
        super().__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf)
        self._ner = ner

    def _mixed_documents(self, raw_documents) -> List[str]:
        """
        Mix documents with ner types - simply insert the ner types
        before raw documents. Example:
            raw docuemnt: "Allen like cake."
            ner results: [
                {'value': 'Allen', 'type': 'person', 'start': 0, 'end': 5},
                {'value': 'cake', 'type': 'food', 'start': 11, 'end': 15}
            ]
            mixed docment: "{person} {food} Allen like cake."


        Parameters
        ----------
        raw_documents: an iterable which yields str

        Returns
        -------
        mixed documents, a list of str

        """
        if not self._ner:
            return raw_documents

        mixed_documents = []
        for doc in raw_documents:
            entities = [
                "{" + entity["type"] + "}" for entity in self._ner.process(doc)]
            if entities:
                mixed_documents.append(" ".join(entities) + " " + doc)
            else:
                mixed_documents.append(doc)

        return mixed_documents

    def fit(self, raw_documents, y=None):
        return super().fit(self._mixed_documents(raw_documents), y)

    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(self._mixed_documents(raw_documents), y)

    def transform(self, raw_documents, copy=True):
        return super().transform(self._mixed_documents(raw_documents), copy)
