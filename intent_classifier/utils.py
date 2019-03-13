"""Class for intent dataset"""

import numpy as np

from sklearn.utils import Bunch


class DataBunch(Bunch):

    def __init__(self, texts: np.array, extended_features: np.array,
                 intents: np.array):
        super().__init__(texts=texts,
                         extended_features=extended_features,
                         intents=intents)


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
    extended_feature - string in json format to offer
                       extended features of context
    intent - string, intent name in form of multi-levels,
             such as "news/sports/football"

    """
    import pymysql

    for key in configs:
        assert key in ["host", "port", "user", "password", "db", "table"]

    texts = []
    efs = []
    intents = []

    db = pymysql.connect(**configs)
    cursor = db.cursor()
    sql = "select text, extended_feature, intent " \
          "from {db}.{table} " \
          "where in_use=1".\
        format(db=configs["db"], table=configs["table"])

    for text, ef, intent in cursor.execute(sql):
        if not intent:
            continue
        if not text and not ef:
            continue
        texts.append(text) if text else texts.append("")
        efs.append(ef) if ef else efs.append("{}")
        intents.append(intent)

    cursor.close()
    db.close()

    return DataBunch(texts=np.array(texts, dtype=np.str),
                     extended_features=np.array(efs, dtype=np.str),
                     intents=np.array(intents, dtype=np.str))
