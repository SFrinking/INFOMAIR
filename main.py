# -*- coding: utf-8 -*-

from classification import Classification

Classification= Classification()
X_train_vectorized, X_test_vectorized,y_train,y_test,vectorizer=Classification.initialize_data()
while True:
    choice = input("Enter 1 for logistic regression, 2 for NN classifier, 3 for SVM, stop to stop: \nEnter 'oversampling' to enable random oversampling on the training set\nEnter 'reset' to split the data again\n")
    if (choice=="oversampling"):
        X_train_vectorized, y_train=Classification.OverSampling()
    if (choice=="reset"):
        Classification.initialize_data()
    if (choice == "1"):
        #classify user utterance
        Classification.TrainLogisticRegression()
        Classification.test_clf()
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                var=Classification.stem_sentence(var).lower()
                x=vectorizer.transform([var])
                y=Classification.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif (choice=="2"):
        Classification.TrainNeuralNetwork()
        Classification.test_clf()
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                var=Classification.stem_sentence(var).lower()
                x=vectorizer.transform([var])
                y=Classification.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif (choice=="3"):
        Classification.TrainSVM()
        Classification.test_clf()
        while True:
            var=input("enter utterance or enter stop to choose another classifier:\n")
            if (var != "stop"):
                var=Classification.stem_sentence(var).lower()
                x=vectorizer.transform([var])
                y=Classification.predict(x)
                print("you have entered a "+y[0]+" utterance")
            else:
                break
    elif(choice=="stop"):
        break
