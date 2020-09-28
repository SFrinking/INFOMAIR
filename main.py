# -*- coding: utf-8 -*-

from classification import Classification
from sklearn.linear_model import LogisticRegression
from baseline import Baseline
from dialogue_agent import Dialogue_Agent
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
da = Dialogue_Agent()


#alternatives(S_test2)
#alternatives(S_test3)

'''
clf_agent=Classification()

clf_agent.open_dataset("dialog_acts.dat")

clf=MLPClassifier()
'''
'''
clf_agent.initialize_data("dialog_acts.dat")
#clf_agent.oversampling()
clf_agent.train_lr()
clf_agent.test_clf()
wrong_preds=clf_agent.get_wrong_predictions()



b = Baseline()
b.open_dataset("dialog_acts.dat")
b.split_dataset()
b.test_keyword_rule()
wrong_pred_baseline=b.get_wrong_predictions()

def make_dict(wrong_preds):
    d={}
    for x, y_pred, y in wrong_preds:
        if (y_pred, y) in d.keys():
            d[(y_pred, y)]+=1
        else:
            d[(y_pred, y)]=1
    return d
w=make_dict(wrong_pred_baseline)
'''
'''
clf_agent.prepare_gs()


params={'learning_rate':['constant'],
        'learning_rate_init':[0.001],
         'solver' : ['adam'],
         'hidden_layer_sizes':[(50),(50,50),(100),(100,100)],
         "max_iter":[200],
         "batch_size":[500]
         }
gs=clf_agent.grid_search(clf, params)
gs.cv_results_
'''

'''
d= da.inference_rules

KB={ "open kitchen","bad hygiene" }

     
    
print(da.make_inferences(KB))
'''

'''
b = Baseline()
b.open_dataset("dialog_acts.dat")
b.split_dataset()
b.get_highest_label()
b.test_highest_label()
print(b.score())
b.test_keyword_rule()
print(b.score())
b.user_input()
#part 1a
'''

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