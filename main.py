# -*- coding: utf-8 -*-

from classification import Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier



#part 1a
'''
clf_agent= Classification()
clf=LogisticRegression(random_state=0,max_iter=200, penalty='l2') 
clf_agent.initialize_data()
scores=agent_clf.cv(clf,True)
'''

#for part 1a
'''
while True:
    choice = input("Enter 1 for logistic regression, 2 for NN classifier, stop to stop: \nEnter 'oversampling' to enable random oversampling on the training set\nEnter 'reset' to split the data again\n")
    if (choice=="oversampling"):
        X_train_vectorized, y_train=agent_clf.OverSampling()
    if (choice=="reset"):
        agent_clf.initialize_data()
    if (choice == "1"):
        #classify user utterance
        agent_clf.TrainLogisticRegression()
        agent_clf.test_clf()
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                var=agent_clf.stem_sentence(var).lower()
                x=vectorizer.transform([var])
                y=agent_clf.clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif (choice=="2"):
        agent_clf.TrainNeuralNetwork()
        agent_clf.test_clf()
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                var=agent_clf.stem_sentence(var).lower()
                x=vectorizer.transform([var])
                y=agent_clf.clf.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif(choice=="stop"):
        break
'''