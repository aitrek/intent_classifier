An implementation of intent classifier based on sklearn for chatbot.


## Introduction  
&nbsp;&nbsp;&nbsp;&nbsp;
The intent recognition is the very key component of a chatbot system. 
We can recognize a man's intent by what a user speak and the dialog context. 
It is a very easy daily activity for us human beings, however, it is a very 
hard task for computers.   

&nbsp;&nbsp;&nbsp;&nbsp;
Here, the intent recognition is treated as a process of multi-labels classification.
Concretely speaking, we use words and context as our input, and the output is 
multi-labels which means a user's words might have multi-intent. 

+ input:  
    + words:  
      Just a string of what user speak.  
      Example: "I wanna known what time is it and how is the weather?"
      
    + context:  
      A json string with kinds of context information.  
      Example: '{"timestamp": 1553154627, "location": "Shanghai"}'
      
  The words and context will be transformed to tfidf-vector and dict-vector 
  respectively, and then the two vectors will be concatenated to form the final
  input vector.

+ intent labels:
    + multi-labels:  
      Labels string separated with ",", such as "time,weather".
     
    + multi-levels:  
      Similar intent labels will be put in the same category and form intent 
      with multi-levels.  
      Example:   
      "news/sport_news/football_news",   
      "news/sport_news/basketball_news",


## Dataset  
&nbsp;&nbsp;&nbsp;&nbsp;
The intent dataset can be in any storage, like mysql database or local csv file. 
Two functions to load dataset from mysql and csv file have been implemented 
in dataset.py. They can be used simply by offering parameters, mysql connection 
configure or csv file path.   
&nbsp;&nbsp;&nbsp;&nbsp;
If intent dataset is put in different storage, you 
could implement function just like utils.load_from_mysql/load_from_csv. Just 
remember that the result of the function should be an instance of DataBunch 
which confined the fields of the dataset:  

```python
def load_from_xxx(xxx_params) -> DataBunch:
    words = []
    contexts = []
    intents = []
    
    # get words, contexts, intents
    ... 
    
    return DataBunch(words=np.array(words, dtype=np.str),
                     contexts=np.array(contexts, dtype=np.str),
                     intents=np.array(intents, dtype=np.str))

```

## Training  
Load intent dataset, create an instance of Intent and then run fit the dataset.   
```python
import os

from intent_classifier import Intent
from intent_classifier.dataset import load_from_mysql


mysql_configs = {"host": "xxx.xxx.xxx.xxx", "port": xxx,
                 "user": "xxxx", "password": "xxxx",
                 "db": "xxxx", "table": "xxxx",
                 "customer": "xxxx"}
data_bunch = load_from_mysql(mysql_configs)

folder = os.path.join(os.getcwd(), "models")

intent = Intent(folder=folder, customer="xxx", lang="en", ner=None, n_jobs=-1)
intent.fit(data_bunch)
intent.dump()
```
Note that the param "ner" in Intent is for named entity recognition, which is 
optional to offer entity information in words.


## Save Models  
After finishing the training, run dump() to save the models with name intent.model 
and the report of the training with name report.txt in the same folder. 
```python
intent.dump()
```


## Load Models  
Run load with specific clf_id or with default clf_id to load the most recent 
models.
```python
intent.load(clf_id="20190321113421")
```

## Predict  
Predict intent labels using predict() with words and contexts. The returned will 
be as list of intent labels.
```python
intent_labels = intent.predict(
    word="I wanna known what time is it and how is the weather?",
    context={"timestamp": 1553154627, "location": "Shanghai"}
)
```


## Requirements  
+ Python 3.7
+ numpy
+ scipy
+ pandas
+ scikit-learn==0.20.3
+ joblib
+ pymysql (optional, for dataset from mysql)
+ jieba (optional, for Chinese tokenization)

## Installation  
pip install -e git+https://github.com/aitrek/intent_classifier.git
