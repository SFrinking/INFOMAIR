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

#function to convert list to dictionary
def Convert(lst): 
    counts = {}
    for i in lst:
      counts[i] = counts.get(i, 0) + 1
    return counts

#count majority in training set
dict=Convert(y_train)
print(dict)
#get highest count in dictionary
def GetHighest(dictionary):
    count=0
    k=""
    for key,value in dictionary.items():
        if value>count:
            count=value
            k=key
    return k,count

highest_label, highest_count=GetHighest(dict)


#classify test set
correct=0
incorrect=0

for y in y_test:
    if y==highest_label:
        correct+=1
    else:
        incorrect+=1
print("accuracy of baseline: " + str(correct/(len(y_test))))
#get user input, loop

import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

keywords_m = {"request": ["where", "what", "whats", "type", "phone", "number",  #meaningful categories
                        "address", "postcode", "post code"], 
            "reqalts": ["how about", "what about", "anything else"],
            #"inform": ["restaurant", "food"],
            "confirm": ["is it", "does it", "do they", "is there", "is that"]
             }
                
    
keywords_ts = {"repeat": ["repeat", "back", "again"],                   #turn-service categories
            "ack": ["okay", "kay", "ok", "fine"],
            "deny": ["wrong", "dont want", "not"],
            "reqmore": ["more"],
            "affirm": ["yes", "right", "correct", "yeah", "uh huh"],
            "negate": ["no"]
              }
            
keywords_ds = {"hello": ["hi", "hello"],                                #dialogue-service categories
            "goodbye": ["goodbye", "good bye", "bye", "stop"],
            "thankyou": ["thank", "thanks"],
            "restart": ["start over", "reset", ],
            "null": ["cough", "unintelligible", "tv_noise", "noise", 
                     "sil", "sigh", "um"]
              }

def keyword_rule(phrase):
    res = "inform"
    
    for key_dict in [keywords_m, keywords_ts, keywords_ds]:
        for k,v in key_dict.items():
            for keyword in v:
                if len(keyword.split(' ')) > 1:
                    if keyword in phrase:
                        res = k
                        break
                else:
                    if keyword in word_tokenize(phrase):
                        res = k
                        break
    return res

#evaluate rule-based classification

allcounter = 0
truecounter = 0

with open("dialog_acts.dat", "r") as infile:

    for line in infile:
        splitlst = line.lower().split(" ", 1)
        res = keyword_rule(splitlst[1])
        allcounter += 1
        if res == splitlst[0]:
            truecounter +=1
            #outfile.write(splitlst[1] + ' ' + splitlst[0] + " TRUE\n" )
        #else:
        #    outfile.write(res + ' ' + splitlst[1] + splitlst[0] + " FALSE\n" )
            
acc = truecounter/allcounter
print("accuracy keyword matching = {}".format(acc))

while True:
    choice = input("press 1 for baseline, 2 for keyword matching:\n")
    if (choice == "1"):
        #classify user utterance
        while True:
            var=input("enter utterance or enter stop to exit:\n")
            if (var != "stop"):
                print("you have entered a "+highest_label+" utterance")
            else:
                break
    elif (choice=="2"):
        while True:
            phrase = input("enter utterance or enter stop to exit:\n")
            if phrase == 'stop':
                break
            else:
                res = keyword_rule(phrase)
                print("You have entered a {} utterance".format(res))
    elif( choice=="stop"):
        break


