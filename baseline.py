def open_dataset(filename):
    """
    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    utterances : list
        X.
    labels : list
        y.

    """
    labels = list()
    utterances = list()
    with open(filename, "r") as infile:
        for line in infile:
            label_utterance = line.lower().split(" ", 1)
            labels.append(label_utterance[0])
            utterances.append(label_utterance[1])
        return utterances,labels



def split_dataset(X,y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X,y, test_size=0.15)
    



def convert_to_dict_freq(lst): 
    """

    Parameters
    ----------
    lst : list
        list of labels.

    Returns
    -------
    counts_dictionary : dict
        k,v where k=label and v=frequency.

    """
    counts_dictionary = {}
    for i in lst:
      counts_dictionary[i] = counts_dictionary.get(i, 0) + 1
    return counts_dictionary


#get most frequent label and its frequency
def get_highest(dictionary):
    count=0
    k=""
    for key,value in dictionary.items():
        if value>count:
            count=value
            k=key
    return k,count



def baseline_one(y_test, highest_label):
    """
    given a list of labels, classify everything in y_test as that label

    Parameters
    ----------
    y_test : list
        
    highest_label : str
        highest label in training set.

    Returns
    -------
    correct, incorrect: int
        how many occurrances correctly and incorrectly classified

    """
    correct=0
    incorrect=0
    
    for y in y_test:
        if y==highest_label:
            correct+=1
        else:
            incorrect+=1
    return correct, incorrect



import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

def get_key_words():
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
    return keywords_m,keywords_ts,keywords_ds 



def keyword_rule(phrase):
    """
    Given a phrase, predict the label based on key-word matching rules

    Parameters
    ----------
    phrase : str
        user input.

    Returns
    -------
    y_pred : str
        prediction of this classifier.

    """
    y_pred = "inform"
    
    for key_dict in [keywords_m, keywords_ts, keywords_ds]:
        for k,v in key_dict.items():
            for keyword in v:
                if len(keyword.split(' ')) > 1:
                    if keyword in phrase:
                        y_pred = k
                        break
                else:
                    if keyword in word_tokenize(phrase):
                        y_pred = k
                        break
    return y_pred



def baseline_two(X):
    """
    predict X based on key-word matching rules

    Parameters
    ----------
    X : list
        list of sentences to predict.

    Returns
    -------
    correct, incorrect: int
        how many occurrances correctly and incorrectly classified

    """
    correct = 0
    incorrect=0
    
    for n in range(len(X)):
        y_pred=keyword_rule(X[n])
        if y_pred == y[n]:
            correct+=1
        else:
            incorrect+=1
    return correct,incorrect
      

def user_utterance(x):
    """
    

    Parameters
    ----------
    x : string or function
        if string, use it as y_pred. If function, call function on phrase.

    Returns
    -------
    y_pred : string
        prediction.

    """
    while True:
        phrase=input("enter utterance or enter stop to exit:\n")
        if (phrase != "stop"):
            y_pred=""
            if callable(x):
                y_pred=x(phrase)
            else:
                y_pred= x
            
            print("You have entered a {} utterance".format(y_pred))
            return y_pred
        else:
            break


X,y= open_dataset("dialog_acts.dat")
X_train, X_test, y_train, y_test = split_dataset(X,y)
while True:
    choice = input("Enter 1 for baseline, 2 for keyword matching, 'stop' to exit:\n")
    if (choice == "1"):
        dict=convert_to_dict_freq(y_train)
        highest_label, highest_count=get_highest(dict)
        correct, incorrect= baseline_one(y_test, highest_label)
        acc = correct/len(y_test)
        misclassification= incorrect/len(y_test)
        print("accuracy,misclassification of keyword matching = {},{}".format(acc,misclassification))
        y_pred=user_utterance(highest_label)
    elif (choice=="2"):
        keywords_m,keywords_ts,keywords_ds = get_key_words()  
        correct,incorrect=baseline_two(X)
        acc = correct/len(X)
        misclassification= incorrect/len(X)
        print("accuracy,misclassification of keyword matching = {},{}".format(acc,misclassification))
        y_pred=user_utterance(keyword_rule)
        
    elif( choice=="stop"):
        break


