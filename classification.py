import numpy as np
def open_dataset(filename):
    """
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

    """
    print('getting data from file '+filename+"...")
    labels = list()
    utterances = list()
    with open(filename, "r") as infile:
        for line in infile:
            label_utterance = line.lower().split(" ", 1)
            labels.append(label_utterance[0])
            utterances.append(label_utterance[1])
        return utterances,labels


def split_dataset(X,y):
    print("splitting dataset into train and test...")
    from sklearn.model_selection import train_test_split
    return train_test_split(X,y, test_size=0.15)
    


#convert list of duplicates into dict with k,v where k=label and v=count
def Convert(lst): 
    counts = {}
    for i in lst:
      counts[i] = counts.get(i, 0) + 1
    return counts

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(X_train,X_test):
    """
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

    """
    print("creating bag of words representation...")
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized=vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, vectorizer


def initialize_data():
    """
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

    """
    global title
    X,y=open_dataset("dialog_acts.dat")
    X_train, X_test, y_train, y_test = split_dataset(X,y)
    title="Without Oversampling"
    X_train_vectorized, X_test_vectorized,vectorizer=bag_of_words(X_train,X_test)
    return X_train_vectorized, X_test_vectorized, y_train,y_test,vectorizer




def OverSampling():
    """
    Perform random oversampling using imblearn.over_sampling. 
    Using as a solution to imbalanced labels.

    Returns
    -------
    X_train_vectorized: sparse matrix
        oversampled dataset.
    y_train: sparse matrix
        oversampled dataset.

    """
    print('Performing oversampling...')
    global title
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42 )
    title="With Oversampling"
    return ros.fit_resample(X_train_vectorized, y_train)
    

#taken from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def plot_confusion_matrix(cm,target_names,y_predicted,title, cmap=None,normalize=False):
    """
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

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    eval_metrics=EvalMetrics(y_predicted,y_test)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\n{}'.format(accuracy, misclass,eval_metrics))
    plt.show()



#plot confusion matrix and print label counts
def make_confusion_matrix(y_predicted, classifier_name):
    """
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

    """
    print("creating confusion matrix...")
    dictionary=Convert(y_test)
    #create confusion matrix and plot
    from sklearn.metrics import confusion_matrix
    labels_no_duplicates= list(dict.fromkeys(dictionary))

    cm=confusion_matrix(y_test,y_predicted, labels=labels_no_duplicates)
    
    plot_confusion_matrix(cm,labels_no_duplicates, y_predicted, classifier_name)
    
#get evaluation metrics
def EvalMetrics(y_predicted,y_test):
    """
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

    """
    
    from sklearn.metrics import recall_score
    mi_recall=recall_score(y_test, y_predicted, average="micro")
    ma_recall=recall_score(y_test, y_predicted, average="macro")
    
    from sklearn.metrics import precision_score
    mi_precision=precision_score(y_test, y_predicted, average='micro')
    ma_precision=precision_score(y_test, y_predicted, average='macro')
    
    from sklearn.metrics import f1_score
    mi_f_score= f1_score(y_test, y_predicted, average='micro')
    ma_f_score= f1_score(y_test, y_predicted, average='macro')
    return {'mi_recall':round(mi_recall,4), 'mi_precision':round(mi_precision,4), 'mi_f_score':round(mi_f_score,4),'ma_recall':round(ma_recall,4), 'ma_precision':round(ma_precision,4), 'ma_f_score':round(ma_f_score,4)}
    


def test_clf(clf,clfName):
    """
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

    """
    print("predicting test set...")
    y_pred=clf.predict(X_test_vectorized)
    make_confusion_matrix(y_pred, title+clfName)
    

def TrainLogisticRegression():
    print('Training LR classifier...')
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0,solver='lbfgs', max_iter=200)
    clf.fit(X_train_vectorized, np.ravel(np.reshape(y_train,(-1,1))))  
    return clf

def TrainNeuralNetwork():
    print('Training NN classifier...')
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=200)
    clf.fit(X_train_vectorized, np.ravel(np.reshape(y_train,(-1,1))))  
    return clf

def TrainSVM():
    print('Training SVM classifier with balanced weights...')
    from sklearn import svm
    clf = svm.SVC(class_weight='balanced')
    clf.fit(X_train_vectorized, np.ravel(np.reshape(y_train,(-1,1))))  
    return clf


title="Without Oversampling"
X_train_vectorized, X_test_vectorized,y_train,y_test,vectorizer=initialize_data()
while True:
    choice = input("Enter 1 for logistic regression, 2 for NN classifier, 3 for SVM, stop to stop: \nEnter 'oversampling' to enable random oversampling on the training set\nEnter 'reset' to split the data again\n")
    if (choice=="oversampling"):
        X_train_vectorized, y_train=OverSampling()
    if (choice=="reset"):
        X_train_vectorized, X_test_vectorized,y_train,y_test, vectorizer=initialize_data()
    if (choice == "1"):
        #classify user utterance
        lr_clf=TrainLogisticRegression()
        test_clf(lr_clf," Logistic Regression")
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                x=vectorizer.transform([var])
                y=lr_clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif (choice=="2"):
        nn_clf=TrainNeuralNetwork()
        test_clf(nn_clf," NN")
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                x=vectorizer.transform([var])
                y=nn_clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif (choice=="3"):
        svm_clf=TrainSVM()
        test_clf(svm_clf," SVM")
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                x=vectorizer.transform([var])
                y=svm_clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif(choice=="stop"):
        break

