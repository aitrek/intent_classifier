"""Class for intent dataset"""

import numpy as np

from typing import Dict, Set

from sklearn.utils import Bunch


class DataBunch(Bunch):

    def __init__(self, words: np.array, contexts: np.array, intents: np.array):
        super().__init__(words=words, contexts=contexts, intents=intents)


def load_from_mysql(configs: dict):
    """
    Load intent dataset from mysql database.

    Parameters
    ----------
    configs: Configs of mysql connnection, which includes keys:
        "host" - host of database,
        "port" - port of database
        "user" - user of database,
        "password" - password to login the database,
        "db" - database name of the dataset,
        "table" - table name of the dataset,
        "charset" - charset of the dataset, default value "utf8".

    Returns
    -------
    Sklearn Bunch instance, including attributes:
    words - strings, user's words
    context - string in json format to offer extended features of context
    intent - string, intent name in form of multi-levels,
             such as "news/sports/football"

    """
    import pymysql

    for key in configs:
        assert key in ["host", "port", "user", "password", "db", "table"]

    words = []
    contexts = []
    intents = []

    db = pymysql.connect(**configs)
    cursor = db.cursor()
    sql = "select word, context, intent " \
          "from {db}.{table} " \
          "where in_use=1".\
        format(db=configs["db"], table=configs["table"])

    for word, context, intent in cursor.execute(sql):
        if not intent:
            continue
        if not word and not context:
            continue
        words.append(word) if word else words.append("")
        contexts.append(context) if context else context.append("{}")
        intents.append(intent)

    cursor.close()
    db.close()

    return DataBunch(words=np.array(words, dtype=np.str),
                     contexts=np.array(contexts, dtype=np.str),
                     intents=np.array(intents, dtype=np.str))


def get_intent_classes(intents) -> Dict[str, Set[str]]:
    """
    Disassemble intents in form of multi-levels to intents of diferent levels.

    Examples:
        Intents:
            [
                "news/sports_news/football",
                "news/sports_news/basketball",
                "news/sports_news/others",
                "news/tech_news",
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
    Disassembled class list.

    """
    assert len(intents) > 0, "intents is empty!"

    cls = {"root": set()}
    for intent in intents:
        levels = intent.split("/")
        name = levels[0]
        cls["root"].add(name)
        for level in levels[1:]:
            if name not in cls:
                cls[name] = set()
            new_name = name + "/" + level
            cls[name].add(new_name)
            name = new_name

    return cls
