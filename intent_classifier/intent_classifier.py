"""Class for intent operations - training, predict"""

import os
import re
import json
import datetime

import joblib
import numpy as np
import pandas as pd

from typing import List, Union

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

from .base import DatasetBunch, RuleBunch, Classifier
from .utils import get_intent_labels, make_dir
from .vectorizer import TfidfVectorizerWithEntity
from .transformer import TextPreprocess, PercentSVD


DEFAULT_FOLDER = os.path.join(os.getcwd(), "models")


class OneClassClassifier(Classifier):
    """Classifier used for dataset which has only one class."""

    def __init__(self, intent: str):
        self._intent = intent

    def predict(self, words: str="", context: Union[str, dict]=None) -> List[str]:
        return [self._intent]


class RuleClassifier(Classifier):
    """Rule-based classifier"""

    def __init__(self, rule_bunch: RuleBunch):
        self._patterns = [re.compile(r) if r else None
                          for r in rule_bunch.word_rules]
        try:
            self._context_rules = \
                [json.loads(r) if r else {} for r in rule_bunch.context_rules] \
                if rule_bunch.context_rules else []
        except AttributeError:
            self._context_rules = []
        self._intent_labels = rule_bunch.intent_labels

    def predict(self, words: str="", context: Union[str, dict]=None) -> List[str]:
        """
        Predict intent labels according to words patterns and comparision
        between context and context_rule.

        Parameters
        ----------
        words: user input
        context: context information

        Returns
        -------
        List of predicted labels or empty list if failed to the matches.

        """
        def context_match(context: dict, rule_context: dict) -> bool:
            if not rule_context:
                return True
            else:
                return False if not context else \
                    all(rule_context.get(k) == v for k, v in context.items())

        # make sure the context to be a dict
        if not context:
            context = {}
        else:
            if isinstance(context, str):
                context = json.loads(context)

        if not words and not context:
            return []

        intent_labels = []
        for i, pattern in enumerate(self._patterns):
            if not self._context_rules:
                if pattern.match(words):
                    for label in self._intent_labels[i]:
                        intent_labels.append(label)
            else:
                if pattern.match(words) and \
                      context_match(context, self._context_rules[i]):
                    for label in self._intent_labels[i]:
                        intent_labels.append(label)

        return intent_labels


class ModelClassifier(Classifier):

    def __init__(self, folder: str=DEFAULT_FOLDER, customer: str="common",
                 lang="en", ner=None, n_jobs=None):
        """

        Parameters
        ----------
        folder: The folder to save the final models.
        customer: Name used to distinguish different customers.
        lang: Language, "en" for English or "cn" for Chinese.
        ner: instance of named entity recognition.
            Its output, taking "Allen like cake." for example,
            should be a list in form:
            [
                {'value': 'Allen', 'type': 'person', 'start': 0, 'end': 5},
                {'value': 'cake', 'type': 'food', 'start': 11, 'end': 15}
            ]
        n_jobs : n_jobs in GridSearchCV, int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        """
        self._folder = folder
        self._customer = customer
        self._lang = lang
        self._ner = ner
        self._n_jobs = n_jobs
        self._classifiers = {}
        self._mlbs = {}
        self._reports = {}

    def fit(self, data_bunch: DatasetBunch):
        """
        Fit with GridSearchCV method to find the optimal parameters.
        Disassemble the intents in form of multi-levels to get sub-datasets
        and train models using these sub-datasets.

        Parameters
        ----------
        data_bunch: Data bunch instance with texts, extended_features, intents.

        """
        def make_choice(labels: str, prefixs: set) -> bool:
            for label in labels.replace(" ", "").split(","):
                for prefix in prefixs:
                    if label.startswith(prefix):
                        return True
            else:
                return False

        def make_labels(labels_data: np.array, label_set: set) -> List[List[str]]:
            labels = []
            for labels_str in labels_data:
                lbls = []
                for label in labels_str.replace(" ", "").split(","):
                    lbls += [lbl for lbl in label_set if label.startswith(lbl)]
                labels.append(lbls)
            return labels

        make_choice_vect = np.vectorize(make_choice)
        for clf_name, label_set in get_intent_labels(data_bunch.intents).items():

            if len(label_set) == 1:
                self._classifiers[clf_name] = \
                    OneClassClassifier(list(label_set)[0])
                self._reports[clf_name] = {"clf_type": "OneClassClassifier"}
            else:
                choices = make_choice_vect(data_bunch.intents, label_set)
                mlb = MultiLabelBinarizer(classes=list(label_set))
                search = self._fit(
                    X=pd.DataFrame({
                        "words": data_bunch.words[choices],
                        "contexts": [json.loads(c) if c else {}
                            for c in data_bunch.contexts[choices]]}),
                    y=mlb.fit_transform(
                        make_labels(data_bunch.intents[choices], label_set))
                )

                self._classifiers[clf_name] = search.best_estimator_
                self._mlbs[clf_name] = mlb

                self._reports[clf_name] = {
                    "clf_type": "sklearn-classifier",
                    "scoring": search.scoring,
                    "cv": search.cv,
                    "best_params": search.best_params_,
                    "best_score": search.best_score_,
                }

    def _fit(self, X: pd.DataFrame, y: np.array):
        """Fit classifier

        Parameters
        ----------
        # X: pd.DataFrame with columns "words" and "contexts".
        X: tuple of "words" and "contexts".
        y: intent labels

        Returns
        -------
        Instance of sklearn classifier or OneClassClassifier.

        """
        def has_context(contexts):
            if contexts.empty:
                return False
            for context in contexts:
                if not context:
                    continue
                if json.loads(context):
                    return True
            else:
                return False

        if has_context(X["contexts"]):
            vectorizer = ColumnTransformer([
                # words to vectors
                ("words2vect",
                 Pipeline([
                     ("text_preprocess", TextPreprocess(self._lang)),
                     ("tfidf_vect", TfidfVectorizerWithEntity(ner=self._ner))
                 ]),

                 "words"),
                # contexts to vectors
                ("contexts2vect", DictVectorizer(), "contexts")
            ])
        else:
            vectorizer = ColumnTransformer([
                # words to vectors
                ("words2vect",
                 Pipeline([
                     ("text_preprocess", TextPreprocess(self._lang)),
                     ("tfidf_vect", TfidfVectorizerWithEntity(ner=self._ner))
                 ]),
                 "words")
            ])

        pipeline = Pipeline([
            # transform words and contexts to vectors
            ("vectorizer", vectorizer),

            # feature values standardization
            ("scaler", StandardScaler(with_mean=False)),

            # dimensionality reduction
            ("svd", PercentSVD()),

            # classifier
            ("clf", RandomForestClassifier())
        ])
        params = {
            "svd__percent": np.linspace(0.1, 1, 3),     # todo
            "clf__n_estimators": range(5, 100, 5),
            "clf__max_features": [None, "sqrt", "log2"],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        }
        search = GridSearchCV(estimator=pipeline, param_grid=params, cv=5,
                              n_jobs=self._n_jobs)
        search.fit(X, y)

        return search

    def predict(self, words: str="", context: Union[str, dict]=None) -> List[str]:
        """

        Parameters
        ----------
        words: user input
        context: context information

        Returns
        -------
        List of predicted labels.

        """
        if not context:
            X = pd.DataFrame({"words": [words], "contexts": ["{}"]})
        else:
            if isinstance(context, str):
                X = pd.DataFrame({"words": [words], "contexts": [context]})
            elif isinstance(context, dict):
                X = pd.DataFrame({"words": [words],
                                  "contexts": [json.dumps(context)]})
            else:
                X = pd.DataFrame({"words": [words], "contexts": ["{}"]})

        return self._predict("root", X)

    def _predict(self, intent: str, X: pd.DataFrame) -> List[str]:
        """
        Predict labels using classifiers and multilabelbinarizers.

        Parameters
        ----------
        intent: intent name
        X: word and context in form of pd.Dataframe

        Returns
        -------
        Tuple of predicted labels.

        """
        intent_labels = []
        if isinstance(self._classifiers[intent], OneClassClassifier):
            intent_labels += self._classifiers[intent].predict(X)
        else:
            for labels in self._mlbs[intent].inverse_transform(
                    self._classifiers[intent].predict(X)):

                for label in labels:
                    if label in self._classifiers:
                        intent_labels += self._predict(label, X)
                    else:
                        intent_labels.append(label)

        return intent_labels

    def _check_id(self, id: str):
        """
        Check if the models' folder exists.

        Parameters
        ----------
        id: IntentClassifier id.

        Returns
        -------
        Bool:
            True - self._folder/id exists
            False - self._folder/id does not exists

        """
        if not id:
            return False
        else:
            return os.path.isfile(
                os.path.join(self._folder, self._customer, id) + ".models")

    def load(self, clf_id: str=None):
        """

        Parameters
        ----------
        clf_id: Classifier id, which comes from the training date time,
            such as "20190313110145". If it is None, the model with maximum
            id will be loaded.

        """
        def max_model_id(model_folder) -> str:
            max_id = 0
            for f in os.listdir(model_folder):
                if not os.path.isdir(os.path.join(model_folder, f)):
                    continue
                else:
                    dir_name = f.split("/")[-1]
                    if dir_name.isdigit():
                        dir_num = int(dir_name)
                        if dir_num > max_id:
                            max_id = dir_num
            return str(max_id)

        clf_dir = os.path.join(self._folder, self._customer)

        assert os.path.isdir(clf_dir), "The model's folder doesn't exists!"

        if clf_id:
            assert os.path.isfile(os.path.join(clf_dir, clf_id, "intent.model")), \
                "clf_id error!"
        else:
            clf_id = max_model_id(clf_dir)

        model = joblib.load((os.path.join(clf_dir, clf_id, "intent.model")))

        self._classifiers = model["clfs"]
        self._mlbs = model["mlbs"]

    def dump(self):
        """
        Save classifiers in self._folder/self._cumstomer/self._id + ".model"
        """
        clf_id = (str(datetime.datetime.now())
                  .replace(" ", "").replace("-", "").replace(":", "")
                  .split("."))[0]
        clf_dir = os.path.join(self._folder, self._customer, clf_id)

        make_dir(clf_dir)

        # save models
        joblib.dump({"clfs": self._classifiers, "mlbs": self._mlbs},
                    os.path.join(clf_dir, "intent.model"))

        # save reports
        with open(os.path.join(clf_dir, "report.txt"), "w") as f:
            json.dump(self._reports, f, indent=4)


class IntentClassifier(Classifier):

    def __init__(self,
                 rule_classifier: RuleClassifier=None,
                 model_classifier: ModelClassifier=None):
        assert rule_classifier is not None or model_classifier is not None
        self._rule_classifier = rule_classifier
        self._model_classifier = model_classifier

    def predict(self, words: str="", context: Union[str, dict]=None) -> List[str]:
        intent_labels = []
        if self._rule_classifier:
            intent_labels = self._rule_classifier.predict(words, context)
        if not intent_labels:
            intent_labels = self._model_classifier.predict(words, context)
        return intent_labels
