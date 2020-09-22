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
# Baseline.py
Implementation of 2 baselines:
1. classify every utterance as majority class
2. classify every utterance based on self-defined rules
Example code:
``` python
    from baseline import Baseline

    b = Baseline()
    b.open_dataset("dialog_acts.dat")
    b.split_dataset()
    b.get_highest_label()
    b.test_highest_label()
    print(b.score())
    b.test_keyword_rule()
    print(b.score())
    b.user_input()
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
``` python
    def __init__(self):
            
            self.X=[]
            self.y=[]
            self.X_train=[]
            self.X_test=[]
            self.y_train=[]
            self.y_test=[]
            self.X_train_vectorized=[]
            self.X_test_vectorized=[]
            self.vectorizer=[]
            self.title=""
            self.clf=""
```
# Functions classification.py
``` python

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