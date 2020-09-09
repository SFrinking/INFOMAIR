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
from sklearn.feature_extraction.text import CountVectorizer

#creating objects
stop_words = set(stopwords.words('english')) 
vectorizer = CountVectorizer(stop_words = stop_words)

X_train_vectorized = vectorizer.fit_transform(X_train)


#classifier 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train_vectorized, np.reshape(y_train,(-1,1)))

X_test_vectorized=vectorizer.transform(X_test)
y_pred=clf.predict(X_test_vectorized)
accuracy=clf.score(X_test_vectorized,y_test)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import recall_score
recall=recall_score(y_test, y_pred, average="micro")

from sklearn.metrics import precision_score
precision=precision_score(y_test, y_pred, average='micro')

from sklearn.metrics import f1_score
f_score= f1_score(y_test, y_pred, average='micro')
print(recall, precision, f_score)