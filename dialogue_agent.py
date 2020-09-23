#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imported librairies:
import pandas as pd
from Levenshtein import distance as dt
from nltk.corpus import stopwords as s_w
from nltk.tokenize import word_tokenize as w_t
import nltk
from classification import Classification
import re
import random

#%%
class Dialogue_Agent():
    
    def __init__(self):
        self.statelog=[]
        
        self.clf_agent= Classification()
        self.clf_agent.initialize_data("dialog_acts.dat")
        self.clf_agent.train_lr()
        # -- Cesar -- preparation for preference extraction
        file=pd.read_csv("restaurant_info.csv")
        
        #extracting columns
        self.restaurant_names=list(file['restaurantname'])
        self.price_range=list(file['pricerange'])
        self.area=list(file['area'])
        self.food_types=list(file['food'])
        self.phone=list(file['phone'])
        self.address=list(file['addr'])
        self.post_code=list(file['postcode'])
        self.suggestions=[]
        self.responses={"Welcome": 
            ["Hello! I can recommend restaurants to you. To start, please enter your request. You can ask for restaurants by area, price range, or food type\n",
             "Hello and welcome to our restaurant system. You can ask for restaurants by area, price range, or food type. To start, please enter your request\n",
             "Hello! You can ask for restaurants by area, price range, or food type. How may I help you?\n"],
        
            "Area":
                ['What part of town do you have in mind?\n'],
            'Price':
                ['What is your desired price range? Cheap, moderate, or expensive?\n',
                 'Would you prefer a cheap, moderate, or expensive restaurant?\n'],
            'Food':
                ["What kind of food would you like? \n",
                "What type of food do you prefer?\n" ],
            
            "AffirmPreferences":
                ['So, you are looking for a restaurant in the {0}, with {1} price range, serving {2} food,  correct?\n'], 
            
            'Answer':
                ["Okay, here is your recommendation: '{}'. Is it fine? \n",
                "I have found a nice restaurant matching your preferences: ‘{}’. Do you like it? \n",
                "I can recommend a good place that fits your criteria: ‘{}’. Is it fine? \n"],
            
            "NoOptions":
                ["Sorry, there are no recommendations matching your demands. Let’s try to search for another restaurant \n",
                "Unfortunately, I couldn’t find a restaurant that matches your expectations. Let’s try to find something else \n"],
                
            'Goodbye':
                ["Thank you for using our restaurant system. Come back! \n",
                "Thank you, I hope I was useful. Do come back! \n"]
                }
            
        self.dialogue("", "init", [0,0,0])
    #%%

    def classification(self,phrase):
        y_pred=self.clf_agent.predict(phrase)
        return y_pred
    
    # %%
    
    
    
    #function that removes stop words e.g Levenshtein_Distance("the",thai") too little
    def stopwords_removal(self,s):
        tk=w_t(s)
        s=[i for i in tk if not i in (s_w.words('english'))]
        s=" ".join(s)
        s = s.lower()
        #print(s)
        s=s.split(" ")
        #print(s)
        return s
    
    
    
    #%%
        
    def no_preference(self,user_ut, p):
        """
        check if user indicated no preference by using keyword matching
    
        Parameters
        ----------
        user_ut : str
            DESCRIPTION.
        p : list
            list of preferences.
    
        Returns
        -------
        p : list
            DESCRIPTION.
    
        """
        if "world food" in user_ut.lower():
            p[2]='any'
        if "international food" in user_ut.lower():
            p[2]='any'
        
        keywords=re.findall( "any\s(\w+)", user_ut.lower())
        if ("area" in keywords):
            p[0]='any'
        if ("price" in keywords):
            p[1]='any'
        if ("food" in keywords):
            p[2]='any'
        return p
    
    # %%
    # -- Cesar -- preference extraction
    
    def preference_extractor(self,user_ut):
        """
        Extract restaurant preference based on user utterance
    
        Parameters
        ----------
        user_ut : str
            user utterance.
    
        Returns
        -------
        None.
    
        """
        
        p=[0 for i in range(3)] #the preferences are stored in a list of three elements, p[0] for the area, p[1] for the price range and p[2] for the food type
        s=self.stopwords_removal(user_ut)
    
        p=self.no_preference(user_ut, p) #check if user indicated no preference
        
        #keyword matching for the area
        for i in s:
            for j in self.area:
                if i == j:
                    p[0] = j
        if(('north' and 'american' in s) and (s.count('north'))>1):
            p[0]=0
        #print(p)
        #keyword matching for the price range
        for i in s:
            for j in self.price_range:
                if i == j:
                    p[1] = j
                    
        #keyword matching for the food type
        for i in s:
            for j in self.food_types:
                if i == j:
                    p[2] = j
                elif ('asian' and 'oriental' in s):
                    p[2]='asian oriental'
                elif ('modern' and 'european' in s):
                    p[2]='modern european'
                elif ('north' and 'american' in s):
                    p[2]='north american'
                    
        #In case the area has been mispelt
        if (p[0] == 0):
            d = {}
            l=[]
            z=['south', 'centre', 'west', 'east', 'north']
            for i in s:
                for j in z:
                    if (dt(i, j)<=1):   
                        d[j] = dt(i, j)
            #print(d)
            for i in d.values():
                l.append(i)
            if len(l)>0:
                k = min(l)
                key_list = list(d.keys())
                val_list = list(d.values())
                if k<=2:
                    p[0] = key_list[val_list.index(k)]
    
        #In case the price range has been mispelt
        if (p[1] == 0):
            d = {}
            l=[]
            d = {}
            l=[]
            for i in s:
                for j in list(set(self.price_range)):
                    if (dt(i, j)<=3):   
                        d[j] = dt(i, j)
            for i in d.values():
                l.append(i)
            if len(l)>0:
                k = min(l)
                key_list = list(d.keys())
                val_list = list(d.values())
                if k<=2:
                    p[1] = key_list[val_list.index(k)]
            for i in s:
                for j in list(set(self.price_range)):
                    if (dt(i, j)<=3):   
                        d[j] = dt(i, j)
            #print(d)
            for i in d.values():
                l.append(i)
            if len(l)>0:
                k = min(l)
                key_list = list(d.keys())
                val_list = list(d.values())
                if k<=2:
                    p[1] = key_list[val_list.index(k)]
        #In case the  food type has been mispelt                
        #thresolds for Levenshtein distances might need to be better tuned for each preference
        if (p[2] == 0):
            d = {}
            l=[]
            for i in s:
                for j in list(set(self.food_types)):
                    if (dt(i, j)<=2):   
                        d[j] = dt(i, j)
                    elif (dt('asian',i)<=2 or dt('oriental',i)<=2 in s):
                        d['asian oriental']=min([dt('asian',i),dt('oriental',i)])
                    elif (dt('modern',i)<=2 or dt('european',i)<=2 in s):
                        d['modern european']=min([dt('modern',i),dt('european',i)])
                    elif (dt('north',i)<=2 or dt('american',i)<=2 in s):
                        if('north' and 'american' in s):
                            d['north american']=min([dt('north',i),dt('american',i)])
            #print(d)
            for i in d.values():
                l.append(i)
            if len(l)>0:
                k = min(l)
                key_list = list(d.keys())
                val_list = list(d.values())
                if k<=3:
                    p[2] = key_list[val_list.index(k)]    
        return p
    
    
    # %%
    # -- Ivan -- look for matches with preferences in the database
    
    def lookup(self,data):
        """
        Look for restaurants in database using user preferences
    
        Parameters
        ----------
        data : list
            list of preferences.
    
        Returns
        -------
        res : list
            list of restaurants.
    
        """
        res = list()
    
        fit_area = set()
        fit_price = set()
        fit_food = set()
        
        if data[0] == "any" or data[0] == 0:
            fit_area = set(range(len(self.area)))
        else:
            for i,a in enumerate(self.area):
                if a == data[0]:
                    fit_area.add(i)
        if data[1] == "any" or data[1] == 0:
            fit_price = set(range(len(self.price_range)))
        else:
            for j,p in enumerate(self.price_range):
                if p == data[1]:
                    fit_price.add(j)
        if data[2] == "any" or data[2] == 0:
            fit_food = set(range(len(self.food_types)))
        else:
            for k,f in enumerate(self.food_types):
                if f == data[2]:
                    fit_food.add(k)
        option_numbers = fit_area.intersection(fit_price, fit_food)
        if option_numbers:
            for i in option_numbers:
                res.append(self.restaurant_names[i])
                
        return res
    
    
    # %%
    # -- Ivan -- finite-state dialogue agent
    
    def agree(self,user_input):
        """
        check whether user agrees or denies
    
        Parameters
        ----------
        user_input : str
            DESCRIPTION.
    
        Returns
        -------
        bool
            true for agree, false for deny.
    
        """
        response = self.classification(user_input)
        if response in ["ack", "affirm"]:
            return True
        elif response in ["deny", "negate"]:
            return False
        else:
            return response

    
    #%%
    def suggest_restaurant(self):
        recommendation=random.choice(self.suggestions)
        self.suggestions.remove(recommendation)
        return input(random.choice(self.responses.get("Answer")).format(recommendation))
    
    #%%
    
    def dialogue(self, user_input, state, user_preferences):
        
       
        self.statelog.append([user_input,state]) #tuple of user utterance and its associated state. We use this to keep track of state jumps.
    
        
        if state == "exit":
            return
        
        if state in ("init", "hello"):
            user_preferences = [0,0,0]

            user_input = input(random.choice(self.responses.get("Welcome")))
            state = self.classification(user_input)
            self.dialogue(user_input, state, user_preferences)
            return
            
        if state in ("inform", "reqalts"):
            extracted_preferences = self.preference_extractor(user_input)
            for i,d in enumerate(user_preferences):
                if d == 0:
                    user_preferences[i] = extracted_preferences[i]
            state="fill_blanks"
            self.suggestions=self.lookup(user_preferences)
            if (len(self.suggestions)==0) or (len(self.suggestions)==1):
                state="answer"
            self.dialogue(user_input, state, user_preferences)
            return 
    
    
        if state == "fill_blanks": #fills in preferences if there is a blank area
            if user_preferences[0] == 0:
                user_input = input(random.choice(self.responses.get("Area")))
                
                state = self.classification(user_input)
                if "area" not in user_input:
                    user_input+=" area"
            elif user_preferences[1] == 0:
                user_input = input(random.choice(self.responses.get("Price")))
                
                state = self.classification(user_input)
                if "price" not in user_input:
                    user_input+=" price"
            elif user_preferences[2] == 0:
                user_input = input(random.choice(self.responses.get("Food")))
                
                state = self.classification(user_input)
                if "food" not in user_input:
                    user_input+=" food"
            else:
                user_input = input(random.choice(self.responses.get("AffirmPreferences")).format(user_preferences[0],user_preferences[1],user_preferences[2]))
                accept = self.agree(user_input)
                if accept is True:
                    self.suggestions = self.lookup(user_preferences)
                    state = "answer"
                elif accept is False:
                    user_input = ""
                    user_preferences = [0,0,0]
                elif accept=="reqalts":
                    user_preferences=[0,0,0]
                else:    
                    state = "accept"
            self.dialogue(user_input, state, user_preferences)
            return
        
        
        if state == "answer": 
            if self.suggestions:      
                user_input=self.suggest_restaurant()
                state = self.classification(user_input)
                
                if state in ["ack", "affirm"]:
                    state = "goodbye"
                elif state in ["reqalts", "reqmore", "deny", "negate"]:
                    
                    state = "answer"
            else:
                user_input = input(random.choice(self.responses.get("NoOptions")))
                #offer alternatives
                user_preferences = [0,0,0]
                state = self.classification(user_input)
            self.dialogue(user_input, state, user_preferences)
            return
            
        if state in ["reqalts","thankyou", "goodbye", "reset"]:
            if state == "reset":
                print("Restarting the dialogue agent...")
            else:
                print(random.choice(self.responses.get("Goodbye")))
                user_input=input("Would you like to finish here or get another recommendations?")
                if (self.classification(user_input) in ("ack","affirm")):
                    state="exit"
                else:
                    state="init"
                
            self.dialogue(user_input, state, user_preferences)
            return
        
        if state == "repeat":
            try:
                user_input = self.statelog[len(self.statelog) - 3][0]
                state = self.statelog[len(self.statelog) - 3][1]
            except IndexError:
                print("Nowhere to go back, starting again")
                state = init
            self.dialogue(user_input, state, user_preferences)
            return
        
    
        else:
            user_input = input("Please repeat")#statelog[len(statelog) + 1][0]
            state = self.statelog[len(self.statelog) - 2][1]
            self.dialogue(user_input, state, user_preferences)
            return
            
    
            
    
    
    # %%
    '''
    
    statelog = list()
    res=[] #list of restaurant recommendations
    if __name__ == "__main__":
        dialogue("", "init", [0,0,0])
        '''