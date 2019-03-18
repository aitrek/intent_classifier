"""Class for intent operations - training, predict"""

import os
import json
import datetime

import joblib
import numpy as np

from typing import Tuple, List

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from .base import DataBunch, OneClassClassifier
from .utils import get_intent_classes
from .vectorizer import TfidfVectorizerWithEntity


DEFAULT_FOLDER = os.path.join(os.getcwd(), "models")


class Intent:

    def __init__(self, customer: str="common", folder: str=None, id: str=None,
                 ner=None):
        """

        Parameters
        ----------
        customer: Name used to distinguish different customers.
        folder: The folder to save the final models, which default value is
            DEFAULT_FOLDER.
        id: Intent id to distinguish different batch of models, which
            default value comes from the training date time, such as
            "20190313110145".
        ner: instance of named entity recognition.
            Its output, taking "Allen like cake." for example,
            should be a list in form:
            [
                {'value': 'Allen', 'type': 'person', 'start': 0, 'end': 5},
                {'value': 'cake', 'type': 'food', 'start': 11, 'end': 15}
            ]
        """
        self._customer = customer
        self._folder = folder if folder else DEFAULT_FOLDER
        self._classifiers = {}
        self._mlbs = {}
        if self._check_id(id):
            self._id = id
            self._load()
        else:
            self._id = (str(datetime.datetime.now())
                        .replace(" ", "").replace("-", "").replace(":", "")
                        .split("."))[0]
        self._ner = ner

    def fit(self, data_bunch: DataBunch):
        """
        Fit with GridSearchCV method to find the optimal parameters.
        Disassemble the intents in form of multi-levels to get sub-datasets
        and train models using these sub-datasets.

        Parameters
        ----------
        data_bunch: Data bunch instance with texts, extended_features, intents.

        """
        intent_classes = get_intent_classes(data_bunch.intents)
        for clf_name, cls in intent_classes.items():
            choices = np.char.startswith(cls, clf_name)
            classes = np.unique(data_bunch.intents[choices])
            # todo report
            if len(classes) == 1:
                self._classifiers[clf_name] = OneClassClassifier(clf_name)
            else:
                mlb = MultiLabelBinarizer(classes=classes)
                self._classifiers[clf_name] = self._fit(
                    X=(data_bunch.words[choices],                   # np.array
                       [json.loads(c) if c else {}
                        for c in data_bunch.contexts[choices]]),    # List[dict]
                    y=mlb.fit_transform(data_bunch.intents[choices])
                )
                self._mlbs[clf_name] = mlb

    def _fit(self, X: Tuple[np.array, List[dict]], y: np.array):
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
        pipeline = Pipeline([
            # transform words and contexts to vectors
            ("vectorizer", ColumnTransformer([
                # words to vectors
                ("words2vect", TfidfVectorizerWithEntity(ner=self._ner), 0),
                # contexts to vectors
                ("contexts2vect", DictVectorizer(), 1)])
            ),

            # feature values standardization
            ("scaler", StandardScaler()),

            # dimensionality reduction
            ("pca", PCA(n_components="mle")),

            # classifier
            ("clf", RandomForestClassifier())
        ])
        params = {
            "clf__n_estimators": range(5, 100, 5),
            "clf__max_features": [None, "sqrt", "log2"],
            "clf__class_weight": ["balanced", "balanced_subsample"]
        }
        search = GridSearchCV(estimator=pipeline, param_grid=params, cv=5)
        search.fit(X, y)

        return search

    def predict(self, word: str="", context: dict=None) -> List[str]:
        """

        Parameters
        ----------
        word
        context

        Returns
        -------
        List of predicted labels.

        """
        return self._predict("root", word, context)

    def _predict(self, intent: str, word: str="", context: dict=None) -> List[str]:
        """
        Predict labels using classifiers and multilabelbinarizers.

        Parameters
        ----------
        intent: intent name
        word: user input
        context: context information

        Returns
        -------
        Tuple of predicted labels.

        """
        intent_labels = []
        if isinstance(self._classifiers[intent], OneClassClassifier):
            intent_labels.append(
                self._classifiers[intent].predict(word, context))
        else:
            for name in self._mlbs[intent].inverse_transform(
                    self._classifiers[intent].predict(word, context)):
                intent_label = intent + "/" + name
                if intent_label in self._classifiers:
                    intent_labels += self._predict(intent_label, word, context)
                else:
                    intent_labels.append(intent_label)

        return intent_labels

    def report(self):
        """
        Create classifiers' reports and save them in self._folder/self._id.
        """
        pass

    def _check_id(self, id: str):
        """
        Check if the models' folder exists.

        Parameters
        ----------
        id: Intent id.

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

    def _load(self):
        """Load classifiers according to initialized parameters"""
        models = joblib.load(
            os.path.join(self._folder, self._customer, self._id) + ".models"
        )
        self._classifiers = models["clfs"]
        self._mlbs = models["mlbs"]

    def _dump(self):
        """
        Save classifiers in self._folder/self._cumstomer/self._id + ".models"
        """
        clf_dir = os.path.join(self._folder, self._customer)
        if not os.path.isdir(clf_dir):
            os.mkdir(clf_dir)
        joblib.dump({"clfs": self._classifiers, "mlbs": self._mlbs},
                    clf_dir + self._id + ".models")
