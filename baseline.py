##import the data

labels = list()
utterances = list()

with open("dialog_acts.dat", "r") as infile:
  for line in infile:
    label_utterance = line.lower().split(" ", 1)
    labels.append(label_utterance[0])
    utterances.append(label_utterance[1])
    
#function to convert list to dictionary
def Convert(lst): 
    counts = {}
    for i in lst:
      counts[i] = counts.get(i, 0) + 1
    return counts

#count majority in training set
dict=Convert(labels)
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

#get user input, loop
while True:
    choice = input("press 1 for baseline, 2 for keyword matching: ")
    if (choice == "1"):
        #classify user utterance
        while True:
            var=input("enter utterance or enter stop to exit: ")
            if (var != "stop"):
                print("you have entered a "+highest_label+" utterance")
            else:
                break
    elif (choice=="2"):
        break
    elif( choice=="stop"):
        break


