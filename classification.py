##import the data

labels = list()
utterances = list()

with open("dialog_acts.dat", "r") as infile:
  for line in infile:
    label_utterance = line.lower().split(" ", 1)
    labels.append(label_utterance[0])
    utterances.append(label_utterance[1])



#split into train and test
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(utterances, labels, test_size=0.15)


##vectorize training data
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize


#creating bag of words representation
from sklearn.feature_extraction.text import CountVectorizer
stop_words = set(stopwords.words('english')) 
vectorizer = CountVectorizer(stop_words = stop_words)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)

#classifier logistic regression

def plot_confusion_matrix(cm,target_names,title='Confusion matrix', cmap=None,normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

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


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def Convert(lst): 
    counts = {}
    for i in lst:
      counts[i] = counts.get(i, 0) + 1
    return counts


def make_confusion_matrix(y_predicted):
    dictionary=Convert(y_test)
    print(dictionary)
    #create confusion matrix and plot
    from sklearn.metrics import confusion_matrix
    labels_no_duplicates= list(dict.fromkeys(labels))
    cm=confusion_matrix(y_test,y_predicted, labels_no_duplicates)
    plot_confusion_matrix(cm,labels_no_duplicates)
    
#get evaluation metrics
def EvalMetrics(y_predicted):
    
    make_confusion_matrix(y_predicted)
    
    from sklearn.metrics import recall_score
    mi_recall=recall_score(y_test, y_predicted, average="micro")
    ma_recall=recall_score(y_test, y_predicted, average="macro")
    
    from sklearn.metrics import precision_score
    mi_precision=precision_score(y_test, y_predicted, average='micro')
    ma_precision=precision_score(y_test, y_predicted, average='macro')
    
    from sklearn.metrics import f1_score
    mi_f_score= f1_score(y_test, y_predicted, average='micro')
    ma_f_score= f1_score(y_test, y_predicted, average='macro')
    print("micro recall, precision and f_score:" ,mi_recall, mi_precision, mi_f_score)
    print("macro recall, precision and f_score:" ,ma_recall, ma_precision, ma_f_score)

def TrainLogisticRegression():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0,solver='lbfgs', max_iter=200).fit(X_train_vectorized, np.reshape(y_train,(-1,1)))
    
    
    y_pred=clf.predict(X_test_vectorized)
    accuracy=clf.score(X_test_vectorized,y_test)
    EvalMetrics(y_pred)
    
    return clf

def TrainNeuralNetwork():
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=500)
    clf.fit(X_train_vectorized, np.reshape(y_train,(-1,1)))
            
    y_pred=clf.predict(X_test_vectorized)
    accuracy=clf.score(X_test_vectorized,y_test)
    
    EvalMetrics(y_pred)
    return clf

#user input
while True:
    choice = input("enter 1 for logistic regression, 2 for NN classifier, stop to stop: ")
    if (choice == "1"):
        #classify user utterance
        lr_clf=TrainLogisticRegression()
        while True:
            var=input("enter utterance or enter stop to exit: ")
            if (var != "stop"):
                x=vectorizer.transform([var])
                y=lr_clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif (choice=="2"):
        nn_clf=TrainNeuralNetwork()
        while True:
            var=input("enter utterance or enter stop to exit: ")
            if (var != "stop"):
                x=vectorizer.transform([var])
                y=nn_clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif(choice=="stop"):
        break

