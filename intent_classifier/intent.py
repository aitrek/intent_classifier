"""Class for intent operations - training, predict"""

import os
import datetime

import joblib
import numpy as np

from typing import Tuple

from sklearn.model_selection import GridSearchCV

from .base import DataBunch, OneClassClassifier
from .utils import get_intent_classes


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
            words = data_bunch.words[choices]
            context = data_bunch.contexts[choices]
            targets = data_bunch.intents[choices]



    def _fit(self, ):

    def predict(self, text: str="", context: dict=None):
        """

        Parameters
        ----------
        text
        context

        Returns
        -------

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
            clf_folder = os.path.join(self._folder, self._customer, id)
            clf_root = os.path.join(clf_folder, "root.clf")
            return os.path.isdir(clf_folder) and os.path.isfile(clf_root)

    def _load(self):
        """Load classifiers according to initialized parameters"""
        self._classifiers = {}

        clf_folder = os.path.join(self._folder, self._customer, self._id)
        start = len(clf_folder)

        for root, _, files in os.walk(clf_folder):
            clf_root = root[start:]
            for file in files:
                clf_path = os.path.join(root, file)
                clf_key = os.path.join(clf_root, ".".join(file.split(".")[:-1]))
                self._classifiers[clf_key] = joblib.load(clf_path)

    def _dump(self):
        """Save classifiers

        To make it easy to examine the classifiers, the classifiers will
        be save in folder and sub-folder according the intent levels.

        Suffix of the classifiers:
        1.sklearn classifier - clf
        2.OneClassClassifier instance - occlf

        Examples:
             Intents:
                news/sports_news/football
                news/sports_news/basketball
                news/sports_news/others
                news/tech_news
                news/social_news
                travel/culture
                travel/food
                shop/clothes
                chat/greeting
                chat/others
                confirm
                unconfirm
                others

             Intents folder structure:
                |===root.clf                # classes: news, travel, shop, chat, confirm, unconfirm, others
                |---news
                    |===news.clf            # classes: sports_news, tech_news, social_news
                    |---sports_news
                        |===sports_news.clf # classes: football, basketball, others
                |---travel
                    |---travel.clf          # classes: culture, food
                |---shop
                    |---clothes.occlf       # one-class: clothes
                |---chat
                    |---chat.clf            # classes: greeting, others

        """
        for key, clf in self._classifiers.items():
            paths = key.split("/")
            if len(paths) == 1:
                clf_dir = os.path.join(self._folder, self._customer, self._id)
                clf_name = paths[0]
            else:
                clf_dir = os.path.join(self._folder, self._customer, self._id,
                                       *paths[:-1])
                clf_name = paths[-1]

            if isinstance(clf, OneClassClassifier):
                clf_name += ".occlf"
            else:
                clf_name += ".clf"

            # create dir if not exists
            if not os.path.isdir(clf_dir):
                os.mkdir(clf_dir)

            joblib.dump(self._classifiers, os.path.join(clf_dir, clf_name))


