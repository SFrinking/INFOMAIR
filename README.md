# Dialog State Agent

Dialog State Agent created for the course 1-GS Methods in AI research (INFOMAIR) 2020-2021 at Utrecht University.
Current functionality is Classification (classification.py, baseline.py) and start dialogue agent.

Run main.py for the text based chatbot for restaurant domain.
classification.py contains functions to train and test classifiers.

# Packages and imports
```
#dialogue_agent.py:
import pandas as pd
from Levenshtein import distance as dt
from nltk.corpus import stopwords as s_w
from nltk.tokenize import word_tokenize as w_t
from classification import Classification
import re
import random
import time

#classification.py
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#baseline.py
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize


```
# baseline.py
Implementation of 2 baselines:
1. classify every utterance as majority class
2. classify every utterance based on self-defined rules
Score for both baselines is based on accuracy. To get the error, output 1-accuracy

Example code:
``` python
    from baseline import Baseline

    b = Baseline()
    b.open_dataset("dialog_acts.dat")
    b.split_dataset()
    #test baseline 1
    b.get_highest_label()
    b.test_highest_label()
    print(b.score())
```
To test the keyword rules, simply run the function:

```python
    #test baseline 2
    b.test_keyword_rule()
    print(b.score())
```

To get the wrongly predicted sentences of the keyword_rule function:

```python
    print(b.get_wrong_predictions())
```

To classify user utterance, simply run the following command:
```python
    b.user_input()
```


# classification.py

Split and preprocess data, train LR or NN classifier on training set and test on test set
Usage: 

```python
    clf=Classification()
    clf.initialize_data("dialog_acts.dat")
    clf.train_lr()#or clf.train_nn()
    clf.test_clf() #to apply to test set
```

Predict a single sentence after training phase
    
```python
    sentence="Hi, I would like to get a suggestion"
    clf.predict(sentence):
```


To get wrongly classified sentences, after testing:

```python
    wrong_preds=clf_agent.get_wrong_predictions()
    print(wrong_preds)

```

Cross Validation. For this function, create a classifier and call the cv function. Second parameter for cv function is a boolean indicating whether or not to oversample.
    
```python
    lr=LogisticRegression(random_state=0, max_iter=200, penalty='l2')
    clf.cv(lr,False) 
```

GridSearch:

```python
    clf_agent=Classification()
    clf_agent.open_dataset("dialog_acts.dat")
    clf=MLPClassifier()
    clf_agent.prepare_gs()
    params={'learning_rate':['constant'],
            'learning_rate_init':[0.01,0.001,0.0001],
             'solver' : ['adam'],
             'hidden_layer_sizes':[(100,100,100)],
             "max_iter":[100]
             }
    gs=clf_agent.grid_search(clf, params)
    gs.cv_results_
```



# dialogue_agent.py

Class to build a dialog agent. Agent works with states, as depicted in the "STD.pdf" file.
The dialogue agent initializes a classifier trained on dialog acts in part 1a and initializes the data from the database "restaurant_info.csv". 

Basic usage: 

```python
    from dialogue_agent import Dialogue_Agent
    #third parameter indicates which machine learning model to use. "nn" for Neural Network, empty string for Logistic Regression 
    da = Dialogue_Agent("dialog_acts.dat","restaurant_info.csv","nn")
    da.start_dialogue()
```

The dialog agent also keeps track of its states. These can be printed with: 

```python
    da.statelog
```


Extra Configurations:

```
    The agent can be configured after starting the dialogue. 
    
    "configure formal" #use formal sentences. 
    "configure informal" #use informal sentences. Standard configuration.
    "configure delay" #put a 0.5s delay on each answer from the system
    "configure no delay" #remove delay. Standard configuration.
           
```
States:
The agent starts in the initialization state and progresses the conversation and changes states to find a suitable restaurant for the user. Some states include "answer" (to suggest restaurants if it finds any) and "fill_blanks" (used to fill the preference slots).
