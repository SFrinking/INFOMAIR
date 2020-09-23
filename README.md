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
3. train and test classifier using one of the classifiers (LR or NN)
```python
    clf.train_lr()#or
    clf.train_nn()
```
4. Testing and getting performance measures:

Test and make confusion matrix
    
```python
    clf.test_clf()
```

Cross Validation. For this function, create a classifier and call the cv function. Second parameter for cv function is a boolean indicating whether or not to oversample.
    
```python
    lr=LogisticRegression(random_state=0, max_iter=200, penalty='l2')
    clf.cv(lr,False) 
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
    da = Dialogue_Agent()
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
```
