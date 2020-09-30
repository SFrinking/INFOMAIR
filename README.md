# Dialog State Agent

Dialog State Agent created for the course 1-GS Methods in AI research (INFOMAIR) 2020-2021 at Utrecht University.
Current functionality is train and test classifier, classify user utterance.

Run Dialogue_Agent.py for the text based chatbot for restaurant domain.
classification.py contains functions to train and test classifiers.

# Packages
```
Pandas
python-Levenshtein
NLTK
re
random
Numpy
SkLearn
imblearn
GridSearchCV


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
    b.get_highest_label()
    b.test_highest_label()
    print(b.score())
```
To test the keyword rules, simply run the function:

```python
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

Class to split data and train and test classifier. 
Usage: 
1. create instance of class Classification()
```python
    clf=Classification()
```
2. initialize data using initialize_data(filename)
```python
    clf.initialize_data("dialog_acts.dat")
```
3. train classifier using one of the classifiers (LR or NN)
```python
    clf.train_lr()#or
    clf.train_nn()
```
4. Testing and getting performance measures:

Test and make confusion matrix. Also saves the wrongly classified sentences in class variable.
    
```python
    clf.test_clf()
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

Predict a single sentence
    
```python
    sentence="Hi, I would like to get a suggestion"
    clf.predict(sentence):
```

# dialogue_agent.py

Class to build a dialog agent. Agent works with states, as depicted in the "state transition diagram.pdf" file.
The dialogue agent initializes a classifier trained on dialog acts in part 1a and initializes the data from the database "restaurant_info.csv". 

Basic usage: 

```python
    from dialogue_agent import Dialogue_Agent
    da = Dialogue_Agent("dialog_acts.dat","restaurant_info.csv")
```

The dialog agent also keeps track of its states. These can be printed with: 

```python
    da.statelog
```

Some important methods for the dialog agent: 

```python
    da.preference_extractor(user_utterance)
    da.dialogue(self, user_input, state, user_preferences) #recursive state transition function. In each state, the agent interacts with the user.
    da.lookup(user_preferences) #find all restaurants matching user preferences.
    alternative_preferences=da.alternative_preferences(self,user_preferences) #used if no restaurants. Get alternative preferences based on set membership to find additional restaurants.
    da.get_alternative_restaurants(self,alternative_preferences) #get all alternative restaurants using the alternative preferences.
    da.ask_extra_preferences(self,user_preferences) #ask user for additional preferences, suggest restaurant and give reason why restaurant was suggested.
    da.make_inferences(self,KB) #given a knowledge base, infer new knowledge based on self defined inference rules.
```

Extra Configurations:

```python
    da = Dialogue_Agent("dialog_acts.dat","restaurant_info.csv")
    da.configure_formality(True) #False=informal
    da.configure_delay(0.5) #configure_delay in seconds. 
    da.start_dialogue()
```
