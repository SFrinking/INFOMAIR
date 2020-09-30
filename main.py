# -*- coding: utf-8 -*-

from classification import Classification
from dialogue_agent import Dialogue_Agent
from sklearn.neural_network import MLPClassifier

da = Dialogue_Agent("dialog_acts.dat","restaurant_info.csv")
da.configure_formality(True)
da.configure_delay(0.5)
da.start_dialogue()

