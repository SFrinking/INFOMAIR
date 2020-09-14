##import the data



labels = list()
utterances = list()
import nltk
nltk.download('averaged_perceptron_tagger')
with open("dialog_acts.dat", "r") as infile:
  for line in infile:
    label_utterance = line.lower().split(" ", 1)
    labels.append(label_utterance[0])
    utterances.append(label_utterance[1])
    
    
    '''
    tagged_sentence = nltk.pos_tag(label_utterance[1].split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'PRP' ]
    if len(edited_sentence) >0:
        labels.append(label_utterance[0])
        utterances.append(' '.join(edited_sentence))

    '''

#split into train and test
import numpy as np
from sklearn.model_selection import train_test_split
print("splitting data into training and test")
X_train, X_test, y_train, y_test = train_test_split(utterances, labels, test_size=0.15)

#convert list of duplicates into dict with k,v where k=label and v=count
def Convert(lst): 
    counts = {}
    for i in lst:
      counts[i] = counts.get(i, 0) + 1
    return counts

'''
#get dict of words and freq, potential use for stopwords removal
words=[]
for u in utterances:
    words+=u.split()
d=Convert(words)
#creating word frequency dict
from operator import itemgetter
stop_words_list=sorted(d.items(), key=itemgetter(1))
stop_words_list.reverse()
'''

'''
##vectorize training data
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
#creating bag of words representation

stop_words = set(stopwords.words('english')) 
'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
print("creating bag of words representation...")
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)

title="Without Oversampling"
def Reset():
    global title
    print("splitting data into training and test...")
    X_train, X_test, y_train, y_test = train_test_split(utterances, labels, test_size=0.15)
    print("creating bag of words representation...")
    title="Without Oversampling"
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized=vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, y_train,y_test



#perform oversampling on dataset
def OverSampling():
    print('Performing oversampling...')
    global X_train_vectorized
    global y_train
    global title
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42 )
    X_train_vectorized, y_train = ros.fit_resample(X_train_vectorized, y_train)
    title="With Oversampling"

#classifier logistic regression

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

    eval_metrics=EvalMetrics(y_predicted)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\n{}'.format(accuracy, misclass,eval_metrics))
    plt.show()



#plot confusion matrix and print label counts
def make_confusion_matrix(y_predicted, classifier_name):
    print("creating confusion matrix...")
    dictionary=Convert(y_test)
    #create confusion matrix and plot
    from sklearn.metrics import confusion_matrix
    labels_no_duplicates= list(dict.fromkeys(labels))

    cm=confusion_matrix(y_test,y_predicted, labels=labels_no_duplicates)
    
    plot_confusion_matrix(cm,labels_no_duplicates, y_predicted, classifier_name)
    
#get evaluation metrics
def EvalMetrics(y_predicted):
    
    
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
    

def TrainLogisticRegression():
    print('Training LR classifier...')
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0,solver='lbfgs', max_iter=200).fit(X_train_vectorized, np.ravel(np.reshape(y_train,(-1,1))))   
    
    print("predicting test set...")
    y_pred=clf.predict(X_test_vectorized)
    accuracy=clf.score(X_test_vectorized,y_test)
    
    make_confusion_matrix(y_pred, title+" Logistic Regression")
    
    return clf

def TrainNeuralNetwork():
    print('Training NN classifier...')
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=200)
    clf.fit(X_train_vectorized, np.ravel(np.reshape(y_train,(-1,1))))
         
    print("predicting test set...")
    y_pred=clf.predict(X_test_vectorized)
    accuracy=clf.score(X_test_vectorized,y_test)
    make_confusion_matrix(y_pred, title+' Neural Net')
    return clf

def TrainSVM():
    print('Training SVM classifier with balanced weights...')
    from sklearn import svm
    clf = svm.SVC(class_weight='balanced')
    clf.fit(X_train_vectorized, y_train)
    
    print("predicting test set...")
    y_pred=clf.predict(X_test_vectorized)
    accuracy=clf.score(X_test_vectorized,y_test)
    make_confusion_matrix(y_pred, title+' SVM')
    return clf

#user input
while True:
    choice = input("Enter 1 for logistic regression, 2 for NN classifier, 3 for SVM, stop to stop: \nEnter 'oversampling' to enable random oversampling on the training set\nEnter 'reset' to split the data again\n")
    if (choice=="oversampling"):
        OverSampling()
    if (choice=="reset"):
        X_train_vectorized, X_test_vectorized,y_train,y_test=Reset()
    if (choice == "1"):
        #classify user utterance
        lr_clf=TrainLogisticRegression()
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
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                x=vectorizer.transform([var])
                y=nn_clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif (choice=="3"):
        nn_clf=TrainSVM()
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                x=vectorizer.transform([var])
                y=nn_clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif(choice=="stop"):
        break

