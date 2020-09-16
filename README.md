# Dialog State Agent

Dialog State Agent created for the course 1-GS Methods in AI research (INFOMAIR) 2020-2021 at Utrecht University.
Current functionality is train and test classifier, classify user utterance.

# Functions
```
def open_dataset(filename):

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    utterances : list
        X.
    labels : list
        y.

    
def split_dataset(X,y):
    splits data using sklearn.model_selection train_test_split
    
    Returns
    -------
    X_train,X_test,y_train,y_test: list


def convert_to_dict_freq(lst): 

    Parameters
    ----------
    lst : list
        list of labels.

    Returns
    -------
    counts_dictionary : dict
        k,v where k=label and v=frequency.


    
def get_highest(dictionary):

    gets highest frequency based on values
    Returns
    -------
    highest_label:str
    highest_count:int
    
def baseline_one(y_test, highest_label):

    given a list of labels, classify everything in y_test as that label

    Parameters
    ----------
    y_test : list
        
    highest_label : str
        highest label in training set.

    Returns
    -------
    correct, incorrect: int
        how many occurrances correctly and incorrectly classified

    
def get_key_words():
    rules for keyword matching
    Returns
    -------
    keywords_m,keywords_ts,keywords_ds : list
        list of labels and some keywords
    
def keyword_rule(phrase):

    Given a phrase, predict the label based on key-word matching rules

    Parameters
    ----------
    phrase : str
        user input.

    Returns
    -------
    y_pred : str
        prediction of this classifier.



def baseline_two(X):

    predict X based on key-word matching rules

    Parameters
    ----------
    X : list
        list of sentences to predict.

    Returns
    -------
    correct, incorrect: int
        how many occurrances correctly and incorrectly classified


    
def user_utterance(x):    

    Parameters
    ----------
    x : string or function
        if string, use it as y_pred. If function, call function on phrase.

    Returns
    -------
    y_pred : string
        prediction.

```