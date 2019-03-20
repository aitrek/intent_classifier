"""Components on datasets"""

import numpy as np

from .base import DataBunch


def load_from_mysql(configs: dict) -> DataBunch:
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
        "charset" - charset of the dataset, default value "utf8",
        "customer" - the customer's name.

    Returns
    -------
    Sklearn Bunch instance, including attributes:
    words - strings, user's words
    context - string in json format to offer extended features of context
    intent_labels - string, intent name in form of multi-levels
        separated with "," for multi-labels, such as
        "news/sports/football,person/story", which means labels
        "news/sports/football" and "person/story".

    """
    import pymysql

    for key in ["host", "port", "user", "password", "db", "table"]:
        assert key in configs, "mysql configs error!"

    words = []
    contexts = []
    intents = []

    db = pymysql.connect(host=configs["host"], port=configs["port"],
                         user=configs["user"], password=configs["password"])
    cursor = db.cursor()
    customer = configs.get("customer")
    if customer and customer != "common":
        sql = "select word, context, intent_labels " \
              "from {db}.{table} " \
              "where in_use=1 and customer in ('common', '{customer}')". \
            format(db=configs["db"], table=configs["table"],
                   customer=customer)
    else:
        sql = "select word, context, intent_labels " \
              "from {db}.{table} " \
              "where in_use=1 and customer='common'". \
            format(db=configs["db"], table=configs["table"])

    cursor.execute(sql)
    for word, context, intent in cursor.fetchall():
        if not intent:
            continue
        if not word and not context:
            continue
        words.append(word.lower()) if word else words.append("")
        contexts.append(context.lower()) if context else contexts.append("{}")
        intents.append(intent.lower())

    cursor.close()
    db.close()

    return DataBunch(words=np.array(words, dtype=np.str),
                     contexts=np.array(contexts, dtype=np.str),
                     intents=np.array(intents, dtype=np.str))