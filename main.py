# -*- coding: utf-8 -*-

from classification import Classification
from dialogue_agent import Dialogue_Agent
from sklearn.neural_network import MLPClassifier


da = Dialogue_Agent("dialog_acts.dat","restaurant_info.csv")
#user_requirements=['bad food']
#da.check_viability({'busy':['long time', 'not romantic'], 'long time':['romantic']}, user_requirements)


da.start_dialogue()
KB=['not hygiene', 'open kitchen', 'spanish']
da.make_inferences(KB)

up=["any","any","spanish"]
da.lookup(up)

from baseline import Baseline

b = Baseline()
b.open_dataset("dialog_acts.dat")

b.test_keyword_rule()
print(b.score())