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
2. train and test classifier using one of the classifiers (LR or NN)
```python
    clf.train_lr()
```
3. Test
    a. Test and make confusion matrix
```python
    clf.train_lr() #or
    clf.train_nn()
```
    b. Cross Validation. For this function, create a classifier and call the cv function
```python
    lr=LogisticRegression(random_state=0, max_iter=200, penalty='l2')
    clf.cv(lr,False)
```
    c. Predict a sentence
```python
    sentence="Hi, I would like to get a suggestion"
    clf.predict(sentence):
```

