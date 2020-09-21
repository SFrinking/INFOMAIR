# Dialog State Agent

Dialog State Agent created for the course 1-GS Methods in AI research (INFOMAIR) 2020-2021 at Utrecht University.
Current functionality is train and test classifier, classify user utterance.


# Functions Baseline.py

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
# classification.py

Class to split data and train and test classifier. 
Usage: 
1. initialize data using initialize_data()
2. train and test classifier using one of the classifiers
3. Test
    a. Cross Validation
    b. Test and make confusion matrix
    c. Predict a sentence
# Attributes classification.py
```
X_train,X_test,y_train,y_test
X_train_vectorized
    sparse matrix
X_test_vectorized
    sparse matrix
vectorizer
    vectorizer trained on the training set, used transform user input into bag of words representation
title
    title of cm plot
clf
    classifier (Logistic Regression, Neural Network, Support Vector Machine)
```
# Functions classification.py
```

 """
 def stem_sentence(self, sentence):
    Use SnowballStemmer from package nltk.stem.snowball to stem words and reduce word type in corpus

    Parameters
    ----------
    sentence : str
        sentence to be stemmed.

    Returns
    -------
    str
        sentence with all words stemmed.

    """
        
def open_dataset(filename):

    Parameters
    ----------
    filename : str
        file to read from.

    Returns
    -------
    utterances : list
        X.
    labels : list
        y.



def bag_of_words(X_train,X_test):
    
    create bag of words representation by vectorizing input space. 
    Train vectorizer on X_train and apply to both X_train and X_test

    Parameters
    ----------
    X_train : list
        list of utterances from training set.
    X_test : list
        list of utterances from test set.

    Returns
    -------
    X_train_vectorized : sparse matrix
        bag of words representation of training set.
    X_test_vectorized : sparse matrix
        bag of words representation of test set.
    vectorizer : CountVectorizer
        vectorizer trained on training set

  
def initialize_data():
    
    get data from file, split it and create bag of words representation

    Returns
    -------
    X_train_vectorized : sparse matrix
        bag of words representation from training set.
    X_test_vectorized :  sparse matrix
        bag of words representation from test set.
    y_train : list
        true labels from training set.
    y_test : list
        true labels from test set.
    vectorizer : CountVectorizer
        vectorizer trained on training set



def OverSampling():
    
    Perform random oversampling using imblearn.over_sampling. 
    Using as a solution to imbalanced labels.

    Returns
    -------
    X_train_vectorized: sparse matrix
        oversampled dataset.
    y_train: sparse matrix
        oversampled dataset.


def plot_confusion_matrix(cm,target_names,y_predicted,title, cmap=None,normalize=False):
    
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    y_predicted:  what the classifier predicted
    classifier_name: for the title of the plot
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    



def make_confusion_matrix(y_predicted, classifier_name):
    
    make a confusion matrix and plot it using pyplot

    Parameters
    ----------
    y_predicted : list
        predicted labels from the classifier.
    classifier_name : str
        used to make the title of the confusion matrix plot.

    Returns
    -------
    None.

    
def EvalMetrics(y_predicted,y_test):
    
    get the micro and macro recall, precision and f score.

    Parameters
    ----------
    y_predicted : list
        predicted labels from the classifier.
    y_test : list
        true labels.

    Returns
    -------
    dict
        evaluation metrics.

    

def test_clf(clf,clfName):
    
    test the previously trained classifier and plot confusion matrix

    Parameters
    ----------
    clf : classifier
        trained classifier.
    clfName : str
        classifier name.

    Returns
    -------
    None.


def TrainLogisticRegression():
    train LR classifier using training set

def TrainNeuralNetwork():
    train NN classifier using training set

def TrainSVM():
    train SVM classifier using training set

"""
def cv(self,clf,oversampling):
    Use 10-fold cross validation to better estimate the evaluation scores of the classifier.
    Random Oversampling set as extra option for the training set of each fold.

    Parameters
    ----------
    clf : classifier
        DESCRIPTION.
    oversampling : bool
        true=oversample training set, false=dont oversample.

    Returns
    -------
    None.

```