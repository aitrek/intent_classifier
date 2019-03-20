"""Class for intent dataset"""

import os
import numpy as np

from typing import Dict, Set


def get_intent_labels(labels_data: np.array) -> Dict[str, Set[str]]:
    """
    Disassemble intent labels in form of multi-levels to labels
    with diferent levels.

    Examples:
        Intent labelss:
            [
                "news/sports_news/football",
                "news/sports_news/basketball",
                "news/sports_news/others",
                "news/tech_news, news/social_news", # multi-labels
                "news/social_news",
                "travel/culture",
                "travel/food",
                "shop/clothes",
                "chat/greeting",
                "chat/others",
                "confirm",
                "unconfirm",
                "others"
            ]

        Disassembled class dict:
            {
                "root": {"news", "travel", "shop", "chat", "confirm", "unconfirm", "others"},
                "news": {"news/sports_news", "news/tech_news", "news/social_news"},
                "news/sports_news": {"news/sports_news/football", "news/sports_news/basketball"},
                "travel": {"travel/culture", "travel/food"},
                "shop": {"shop/clothes"},
                "chat": {"chat/greeting", "chat/others"},
            }

    Parameters
    ----------
    intents: array-like intent strings

    Returns
    -------
    Disassembled class labels.

    """
    assert len(labels_data) > 0, "intents is empty!"

    intent_labels = {"root": set()}
    for labels_str in labels_data:
        for label in labels_str.replace(" ", "").split(","):
            levels = label.split("/")
            name = levels[0]
            intent_labels["root"].add(name)
            for level in levels[1:]:
                if name not in intent_labels:
                    intent_labels[name] = set()
                new_name = name + "/" + level
                intent_labels[name].add(new_name)
                name = new_name

    return intent_labels


def make_dir(abs_path: str):
    """
    Create a directory with deep path.

    Parameters
    ----------
    abs_path: absolute path of directory.

    """
    dir_path = "/"
    for path in abs_path.split("/"):
        dir_path = os.path.join(dir_path, path)
        if os.path.isdir(dir_path):
            continue
        else:
            os.mkdir(dir_path)
