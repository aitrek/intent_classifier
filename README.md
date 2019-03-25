# An Intent Classifier For Chatbot


## Introduction  
&nbsp;&nbsp;&nbsp;&nbsp;
The intent recognition is the very key component of a chatbot system. 
We can recognize a man's intent by what a user speak and the dialog context. 
It is a very easy daily activity for us human beings, however, it is a very 
hard task for computers.   

&nbsp;&nbsp;&nbsp;&nbsp;
The intent recognition is treated as a process of multi-labels classification.
Concretely speaking, we use words and context as our input, and the output is 
multi-labels which means a user's words might have multi-intent. 

+ input:  
    + words:  
      Just a string of what user speak.  
      Example: "I wanna known what time is it and how is the weather?"
      
    + context(optional):  
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
### Intent Dataset
&nbsp;&nbsp;&nbsp;&nbsp;
The intent dataset can be in any storage, like mysql database or local csv file. 
Two functions to load dataset from mysql and csv file have been implemented 
in dataset.py. They can be used simply by offering parameters, mysql connection 
configure or csv file path.   
&nbsp;&nbsp;&nbsp;&nbsp;
If intent dataset is put in different storage, you could implement function 
just like utils.load_from_mysql/load_from_csv. Just remember that the result 
of the function should be an instance of DataBunch which confined the fields 
of the dataset:  

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
### Rules
&nbsp;&nbsp;&nbsp;&nbsp;
Rules is essentially a kind of dataset just like intent dataset. They can also be 
stored in any storage and fetched in the same way with intent dataset:
```python
def load_from_xxx(xxx_params) -> DataBunch:
    words_rules = []
    context_rules = []
    intent_labels = []
    
    # get words, contexts, intents
    ... 
    
    return RuleBunch(words_rules=words_rules, context_rules=context_rules,
                     intent_labels=intent_labels)

```

## Classifiers
The classification mechanism consists of rule-based and model-based approaches:

* rule-based classification:  
The rule-based approach predict intent labels using hand-written regexps and 
context keys&values to match the words and context. Unlike model-based 
classifier, we don't need to train the rule-based classifier but load rules 
from storage and then create instance directly with the rules:
```python
import os

from intent_classifier import RuleClassifier, ModelClassifier, IntentClassifier
from intent_classifier.dataset import load_intents_from_mysql, load_rules_from_mysql

configs_mysql_rule = {"host": "xxx.xxx.xxx.xxx", "port": xxx,
                      "user": "xxxx", "password": "xxxx",
                      "db": "xxxx", "table": "xxxx",
                      "customer": "xxxx"}
rule_bunch = load_rules_from_mysql(configs_mysql_rule)

rule_classifier = RuleClassifier(rule_bunch)
```

* model-based classification:  
The model-based approach predict intent labels using machine learning models 
trainded from intent dataset. Suppose that we have trained and dumped models, 
we can use the models in this way:
```python
folder = "xxx/xxx/xxx"
model_classifier = ModelClassifier(folder=folder, customer="xxx", lang="en", 
                                   ner=None, n_jobs=-1)
model_classifier.load(clf_id=None)  # load models with maximum id
```

* IntentClassifier:  
IntentClassifier wraps RuleClassifier and ModelClassifier to offer a final
integration interface to predict intent labels. We can use just RuleClassifier 
or ModelClassifier to initialize IntentClassifer or use both them. If the two 
classifiers are in use, the ModelClassifier might be skipped if we have already 
got intent labels from the RuleClassifier.   
Please note that we should try not use to many rules to predict the intent 
labels, the best way of which is to use model-based classifier. The rule-based 
classifier is an option only for very simple case no need of model or as a 
temporary solution when model is not ready, for instance we need retrain the 
model to add some new intents.


## Training  
Load intent dataset, create an instance of ModelClassifier and then fit 
the dataset. 
```python

configs_mysql_model = {"host": "xxx.xxx.xxx.xxx", "port": xxx,
                       "user": "xxxx", "password": "xxxx",
                       "db": "xxxx", "table": "xxxx",
                       "customer": "xxxx"}
data_bunch = load_intents_from_mysql(configs_mysql_model)

folder = os.path.join(os.getcwd(), "models")
model_classifier = ModelClassifier(folder=folder, customer="xxx", lang="en", 
                                   ner=None, n_jobs=-1)
model_classifier.fit(data_bunch)
```
Note that the param "ner" in Intent is for named entity recognition, which is 
optional to offer entity information in words.


## Save Models  
After finishing the fitting, run dump() to save the models in a sub-folder 
with name from datatime in the specified model folder. The models and report 
will be save in the sub-folder with name "intent.model" and "report.txt" 
respectively. 
```python
model_classifier.dump()
```


## Load Models  
Run load with specific clf_id or with default clf_id to load the most recent 
models.
```python
model_classifier.load(clf_id="20190321113421")  # if clf_id is None, the model 
                                                # with maximum id will be loaded
```

## Create IntentClassifier
```python
intent_classifier = IntentClassifier(rule_classifier=rule_classifier,
                                     model_classifier=model_classifier)
```


## Predict  
Predict intent labels using predict() with words and contexts. The returned will 
be as list of intent labels.
```python
intent_labels = intent_classifier.predict(
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
