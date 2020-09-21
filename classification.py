import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
class Classification():
    
    #%%
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
        
    #%%
    def stem_sentence(self, sentence):
        """
        Use SnowballStemmer from package nltk.stem.snowball to stem words and reduce word type in corpus

        Parameters
        ----------
        sentence : str
            DESCRIPTION.

        Returns
        -------
        str
            sentence with all words stemmed.

        """
        stemmer=SnowballStemmer("english")
        stemmed= [stemmer.stem(x) for x in sentence.split()]
        return " ".join(stemmed)
    #%%
    def open_dataset(self,filename):
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
                
                utterances.append(self.stem_sentence(label_utterance[1]))
            #print(utterances)
            return utterances,labels
    
    #%%
    def split_dataset(self,X,y):
        print("splitting dataset into train and test...")
        
        return train_test_split(X,y, test_size=0.15)
        
    
    #%%
    #convert list of duplicates into dict with k,v where k=label and v=count
    def Convert(self,lst): 
        counts = {}
        for i in lst:
          counts[i] = counts.get(i, 0) + 1
        return counts
    
    
    
    #%%
    def bag_of_words(self,X_train,X_test):
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
    
    #%%
    def initialize_data(self):
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
        self.title="Without Oversampling"
        self.X,self.y=self.open_dataset("dialog_acts.dat")
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset(self.X,self.y)
        self.X_train_vectorized, self.X_test_vectorized,self.vectorizer=self.bag_of_words(self.X_train,self.X_test)
        
        return self.X_train_vectorized, self.X_test_vectorized, self.y_train,self.y_test,self.vectorizer
    
    
    
    #%%
    def OverSampling(self):
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
        
        ros = RandomOverSampler(random_state=42 )
        self.title="With Oversampling"
        self.X_train_vectorized, self.y_train=ros.fit_resample(self.X_train_vectorized, self.y_train)
        return self.X_train_vectorized, self.y_train
        
    #%%
    #taken from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    def plot_confusion_matrix(self,cm,target_names,y_predicted,title, cmap=None,normalize=False):
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
    
        eval_metrics=self.EvalMetrics(y_predicted,self.y_test)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\n{}'.format(accuracy, misclass,eval_metrics))
        plt.show()
    
    
    #%%
    #plot confusion matrix and print label counts
    def make_confusion_matrix(self,y_predicted, classifier_name):
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
        dictionary=self.Convert(self.y_test)
        #create confusion matrix and plot
        
        labels_no_duplicates= list(dict.fromkeys(dictionary))
    
        cm=confusion_matrix(self.y_test,y_predicted, labels=labels_no_duplicates)
        
        self.plot_confusion_matrix(cm,labels_no_duplicates, y_predicted, classifier_name)
    #%%   
    #get evaluation metrics
    def EvalMetrics(self,y_predicted,y_test):
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
        
        
        mi_recall=recall_score(y_test, y_predicted, average="micro")
        ma_recall=recall_score(y_test, y_predicted, average="macro")
        
        
        mi_precision=precision_score(y_test, y_predicted, average='micro')
        ma_precision=precision_score(y_test, y_predicted, average='macro')
        
        
        mi_f_score= f1_score(y_test, y_predicted, average='micro')
        ma_f_score= f1_score(y_test, y_predicted, average='macro')
        return {'mi_recall':round(mi_recall,4), 'mi_precision':round(mi_precision,4), 'mi_f_score':round(mi_f_score,4),'ma_recall':round(ma_recall,4), 'ma_precision':round(ma_precision,4), 'ma_f_score':round(ma_f_score,4)}
        
    
    #%%
    def test_clf(self):
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
        y_pred=self.clf.predict(self.X_test_vectorized)
        self.make_confusion_matrix(y_pred, self.title+" "+str(self.clf).split("(")[0])
   

    #%%
    def predict(self, phrase):
        if phrase=="exit":
            return "exit"
        X=self.vectorizer.transform([phrase])
        return self.clf.predict(X)
        
        
    #%%
    def TrainLogisticRegression(self):
        print('Training LR classifier...')
        
        self.clf = LogisticRegression(random_state=0,solver='saga', max_iter=200, penalty='l1')
        self.clf.fit(self.X_train_vectorized, np.ravel(np.reshape(self.y_train,(-1,1))))
        
    
    #%%
    def TrainNeuralNetwork(self):
        print('Training NN classifier...')
        
        self.clf = MLPClassifier(random_state=1, max_iter=200)
        self.clf.fit(self.X_train_vectorized, np.ravel(np.reshape(self.y_train,(-1,1))))  
        
        
    #%%
    def cv(self,clf,oversampling):
        """
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

        """
        
       
        v=CountVectorizer()
        X_vectorized=v.fit_transform(self.X)
        
        kf = KFold(n_splits=10)
        
        accuracy=[]
        f1_macro=[]
        f1_micro=[]
        
        for fold, (train_index, test_index) in enumerate(kf.split(self.X), 1):
            print("fold ",fold)
            X_train = X_vectorized[train_index]
            y_train = np.ravel(np.reshape(self.y,(-1,1)))[train_index]  # Based on your code, you might need a ravel call here, but I would look into how you're generating your y
            X_test = X_vectorized[test_index]
            y_test = np.ravel(np.reshape(self.y,(-1,1)))[test_index]  # See comment on ravel and  y_train
            if oversampling==True:
                ros = RandomOverSampler(random_state=42 )
                X_train,y_train=ros.fit_resample(X_train, y_train)
                
            
            model =  clf# Choose a model here
            model.fit(X_train, y_train)  
            y_pred = model.predict(X_test)
            accuracy.append(model.score(X_test, y_test))
            f1_macro.append(f1_score(y_test, y_pred, average="macro"))
            f1_micro.append(f1_score(y_test, y_pred, average="micro"))
            
            #print(f'For fold {fold}:')
            #print(f"Accuracy: {model.score(X_test, y_test)}")
            #print(f'f-score: {f1_score(y_test, y_pred, average="macro")}')
        print("accuracy: ", sum(accuracy)/kf.n_splits,"f1_macro: ",sum(f1_macro)/kf.n_splits,"f1_micro: ",sum(f1_micro)/kf.n_splits )
        


