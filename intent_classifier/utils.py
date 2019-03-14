"""Class for intent dataset"""

import numpy as np

from sklearn.utils import Bunch


class DataBunch(Bunch):

    def __init__(self, texts: np.array, contexts: np.array, intents: np.array):
        super().__init__(texts=texts, contexts=contexts, intents=intents)


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
    text - string sentences
    context - string in json format to offer extended features of context
    intent - string, intent name in form of multi-levels,
             such as "news/sports/football"

    """
    import pymysql

    for key in configs:
        assert key in ["host", "port", "user", "password", "db", "table"]

    texts = []
    contexts = []
    intents = []

    db = pymysql.connect(**configs)
    cursor = db.cursor()
    sql = "select text, context, intent " \
          "from {db}.{table} " \
          "where in_use=1".\
        format(db=configs["db"], table=configs["table"])

    for text, context, intent in cursor.execute(sql):
        if not intent:
            continue
        if not text and not context:
            continue
        texts.append(text) if text else texts.append("")
        contexts.append(context) if context else context.append("{}")
        intents.append(intent)

    cursor.close()
    db.close()

    return DataBunch(texts=np.array(texts, dtype=np.str),
                     contexts=np.array(contexts, dtype=np.str),
                     intents=np.array(intents, dtype=np.str))
