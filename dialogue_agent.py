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
        self.good_food=list(file['good food'])
        self.open_kitchen=list(file['open kitchen'])
        self.hygiene=list(file['hygiene'])
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
                ['So, you are looking for a restaurant in the {0} part of town, with {1} price range, serving {2} food, correct?\n'], 
            
            'Answer':
                ["Okay, here is your recommendation: '{}'. Is it fine? \n",
                "I have found a nice restaurant matching your preferences: ‘{}’. Do you like it? \n",
                "I can recommend a good place that fits your criteria: ‘{}’. Is it fine? \n"],
            
            "NoOptions":
                ["Sorry, there are no recommendations matching your demands.Let's try finding something else\n",
                "Unfortunately, I couldn’t find a restaurant that matches your expectations. Let’s try to find something else \n"],
                
            'Goodbye':
                ["Thank you for using our restaurant system. Come back! \n",
                "Thank you, I hope I was useful. Do come back! \n"]
                }
            
        self.inference_rules={ "cheap,good food":["busy"],
            "spanish":["longtime"], 
            'busy':['longtime','not romantic'], 
            'long time':['not children', 'not romantic'],  
            'bad hygiene,open kitchen':['not romantic'],
            'bad food,bad hygiene':['not busy'],
            'open kitchen':['children'],
            'long time, not open kitchen':['boring'],
            'boring,expensive':['not busy'],
            'boring':['not romantic']
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
    
    
    
   
    
    # %%
    # -- Cesar -- preference extraction
    
    def preference_extractor(self,user_input):
        """
        Extract restaurant preference based on user utterance
    
        Parameters
        ----------
        user_input : str
            user utterance.
    
        Returns
        -------
        None.
    
        """
        
        p=[0 for i in range(3)] #the preferences are stored in a list of three elements, p[0] for the area, p[1] for the price range and p[2] for the food type
        s=self.stopwords_removal(user_input)
    
        p=self.no_preference(user_input, p) #check if user indicated no preference
        
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
    #%%
    def alternative_preferences(self,S):
        #The input for this function is list S of composed of 3 strings equivalent to user preferences
        #S[0], S[1] and S[2] respectively store area, price range and food type
        a_1=list({'centre', 'north', 'west'})
        a_2=list({'centre', 'north', 'east'})
        a_3=list({'centre', 'south', 'west'})
        a_4=list({'centre', 'south', 'east'})
        north_list=list({'north'})
        a=[a_1,a_2,a_3,a_4]
        
        
        
        #Price range sub-sets
        
        p_1=list({'cheap', 'moderate'})
        p_2=list({'moderate', 'expensive'})
        p=[p_1,p_2]
        
        
        #Food type set & sub-sets
        
        
        f_1=list({'thai', 'chinese', 'korean', 'vietnamese','asian oriental'})
        f_2=list({'mediterranean', 'spanish', 'portuguese', 'italian', 'romanian', 'tuscan', 'catalan'})
        f_3=list({'french', 'european', 'bistro', 'swiss', 'gastropub', 'traditional'})
        f_4=list({'north american', 'steakhouse', 'british'})
        f_5=list({'lebanese', 'turkish', 'persian'})
        f_6=list({'international', 'modern european', 'fusion'})
        f=[f_1,f_2,f_3,f_4,f_5,f_6]
        #Retrieving the criterias
        s_1=S[0]
        s_2=S[1]
        s_3=S[2]
        
        #Retrieving affiliated subset of area s_1
        k=[]
        for i in range(len(a)):
            for j in range(len(a[i])):
                if s_1 in a[i]:
                  k.extend(a[i])
        k_2=list(set(k))
        del k_2[k_2.index(s_1)]  
    
        #price
        l=[]
        for i in range(len(p)):
            for j in range(len(p[i])):
                if s_2 in p[i]:
                    l.extend(p[i])
        l_2=list(set(l))  #remove pairs
        del l_2[l_2.index(s_2)]
        #food
        m=[]
        for i in range(len(f)):
            for j in range(len(f[i])):
                if s_3 in f[i]:
                    m=f[i] #no possible intersections within these sets
        del m[m.index(s_3)]        
        return k_2,l_2,m #output 3 lists but might need to change the type to satisfy the rest of the code
        
     #%%
        
    def no_preference(self,user_input, p):
        """
        check if user indicated no preference by using keyword matching
    
        Parameters
        ----------
        user_input : str
            DESCRIPTION.
        p : list
            list of preferences.
    
        Returns
        -------
        p : list
            DESCRIPTION.
    
        """
        if "world food" in user_input.lower():
            p[2]='any'
        if "international food" in user_input.lower():
            p[2]='any'
        
        keywords=re.findall( "any\s(\w+)", user_input.lower())
        if ("area" in keywords):
            p[0]='any'
        if ("price" in keywords):
            p[1]='any'
        if ("food" in keywords):
            p[2]='any'
        return p
 
 
    #%%
    def grounding(self, user_preferences):
        #the preferences are stored in a list of three elements, p[0] for the area, p[1] for the price range and p[2] for the food type
        answer_template= "So you would like me to find a restaurant "
        p=user_preferences
        if p[0]:
            answer_template+="in the {} of town ".format(p[0])
        if p[1]:
            answer_template+="priced {}ly ".format(p[1])
        if p[2]:
            answer_template+="serving {} cuisine ".format(p[2])
        return answer_template.rstrip()+". "
        
    #%%
    
    def dialogue(self, user_input, state, user_preferences):
        
        self.statelog.append([user_input,state]) #tuple of user utterance and its associated state. We use this to keep track of state jumps.
    
        
        if state == "exit":
            print(random.choice(self.responses.get("Goodbye")))
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
            state="fill_blanks" #if more slots to be filled
            self.suggestions=self.lookup(user_preferences)
            if (len(self.suggestions)==0) or (len(self.suggestions)==1):
                state="answer" #if there is none or 1 restaurant to suggest
            self.dialogue(user_input, state, user_preferences)
            return 
    
        
        
        if state == "fill_blanks": #fills in preferences if there is a blank area
            grounding=self.grounding(user_preferences)
            if user_preferences[0] == 0:
                user_input = input(grounding+random.choice(self.responses.get("Area")))
                
                state = self.classification(user_input)
                if "area" not in user_input:
                    user_input+=" area"
            elif user_preferences[1] == 0:
                user_input = input(grounding+random.choice(self.responses.get("Price")))
                
                state = self.classification(user_input)
                if "price" not in user_input:
                    user_input+=" price"
            elif user_preferences[2] == 0:
                user_input = input(grounding+random.choice(self.responses.get("Food")))
                
                state = self.classification(user_input)
                if "food" not in user_input:
                    user_input+=" food"
         
            else:
                state='ask_extra_preferences'
                
            self.dialogue(user_input, state, user_preferences)
            return
        if state== 'ask_extra_preferences':
            state=self.ask_extra_preferences(user_preferences)
            self.dialogue(user_input, state, user_preferences)
            return
        if state=="confirmpreferences":
            user_input = input(random.choice(self.responses.get("AffirmPreferences")).format(user_preferences[0],user_preferences[1],user_preferences[2]))
            accept = self.agree(user_input)
            if accept is True:
                self.suggestions = self.lookup(user_preferences)
                state = "answer"
            elif accept is False:
                state = "inform"
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
                
                user_input=input(self.suggest_restaurant())
                state = self.classification(user_input)
                
                if state in ["ack", "affirm"]:
                    state = "goodbye"
                elif state in ["reqalts", "reqmore", "deny", "negate"]:
                    
                    state = "answer"
            else:
                
                
                alternatives=self.get_alternative_restaurants(self.alternative_preferences(user_preferences))#offer alternatives
                print(alternatives)
                print(random.choice(self.responses.get("NoOptions"))+"Here is a list of alternatives:")
                for a in alternatives:
                    print(self.get_restaurant_info(a))
                user_input = input('Would you like to choose one (1) or change your preferences(2)?')
                if user_input=="1":
                    user_input=input("Which one would you like to choose?")
                    if user_input in alternatives:
                        self.recommendation=user_input
                        state="thankyou"
                        
                elif user_input=="2":
                    user_preferences=[0,0,0]
                    state='inform'
                
            self.dialogue(user_input, state, user_preferences)
            return
            
        if state in ["reqalts","thankyou", "goodbye", "reset"]:
            if state == "reset":
                print("Restarting the dialogue agent...")
            else:
                user_input=input(self.get_restaurant_contacts(self.recommendation)+". Would you like to finish here?")

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
                print("Nowhere to go back, starting again\n")
                state = "init"
            self.dialogue(user_input, state, user_preferences)
            return
        
    
        else:
            user_input = input("Please repeat")#statelog[len(statelog) + 1][0]
            state = self.statelog[len(self.statelog) - 2][1]
            self.dialogue(user_input, state, user_preferences)
            return
            
        
        
        #%%
    def get_alternative_restaurants(self,alternative_preferences):
        import itertools
        all_alternative_pref=[]
        all_alternative_restaurants=[]
        for r in itertools.product(alternative_preferences[0], alternative_preferences[1],alternative_preferences[2]): 
            all_alternative_pref.append([r[0], r[1],r[2]])
        for a in all_alternative_pref:
            all_alternative_restaurants.append(self.lookup(a))
        all_alternative_restaurants = [item for sublist in all_alternative_restaurants for item in sublist]

        return all_alternative_restaurants
        
    #%%
    def get_user_extra_preferences(self,requirement_options,user_input):
        user_requirements=[]
        for requirement in requirement_options:
            if requirement in user_input:
                if "no "+requirement in user_input or "not "+requirement in user_input:
                    user_requirements.append("not "+requirement)
                else:
                    user_requirements.append(requirement)
        return user_requirements
        #%%
    def ask_extra_preferences(self,user_preferences):
        
        state='confirmpreferences'
        user_input = input("Any other requirements?\n")
        requirement_options=['good food','open kitchen','hygiene', 'children', 'romantic','busy','boring' ]
        user_requirements=self.get_user_extra_preferences(requirement_options,user_input)#extra prefs from user
        
        
        
        print(user_requirements)
        self.suggestions = self.lookup(user_preferences)
        print(self.suggestions)
        for restaurant in self.suggestions:
            i=self.restaurant_names.index(restaurant)
            restaurant_extra_preferences={self.good_food[i], self.open_kitchen[i],self.hygiene[i]}
            applied_rules,KB=self.make_inferences(restaurant_extra_preferences)
            print(applied_rules,KB)
            matching_rules= []
            for user_requirement in user_requirements:
                if (user_requirement in KB):
                    matching_rules.append(user_requirement)
                for k,v in applied_rules.items():
                    if user_requirement==v[0]:
                        matching_rules.append([k,v])
            if (matching_rules):
                user_input=input("{}, this restaurant is recommended because of rule {}. Would you like to choose this restaurant?".format(restaurant,matching_rules))
                answer=self.classification(user_input)
                if answer in ["affirm", "ack"]:
                    self.recommendation=restaurant
                    state="thankyou"
                    return state
                            
                        
        return state
    #%%
    def get_restaurant_info(self, restaurant_name):
        index=self.restaurant_names.index(restaurant_name)
        
        return "Restaurant '{}' serving {} food in {} part of town for {} price".format(restaurant_name.capitalize(),self.food_types[index], self.area[index], self.price_range[index])
    #%%
    def get_restaurant_contacts(self,recommendation):
        i=self.restaurant_names.index(recommendation)#get index of recommendation
        phone=self.phone[i]
        address=self.address[i]
        return "Alright, here are the contacts:'{}', {}, {}".format(recommendation,phone,address)
        
       #%%
    def make_inferences(self,KB):
        """
        Add knowledge to knowledge base KB by making use of inference rules.

        Parameters
        ----------
        KB : set
            convert to list first.

        Returns
        -------
        KB
            as a set, to eliminate duplicates.

        """
        applied_rules={}
        KB=list(KB)
        for knowledge in KB:
            for key,values in self.inference_rules.items():
                if knowledge == key:
                    for v in values:
                        applied_rules[key]=values
                        KB.append(v)
                elif knowledge in key:
                    atoms=key.split(",")
                    if (set(atoms) & set(KB) == set(atoms)):
                        applied_rules[key]=values
                        KB.extend(values)
        return applied_rules,set(KB)     
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
        answer=""
        if len(self.suggestions)==1:
            answer="I could only find one option for you: {}. Is this fine?"
        else:
            answer=random.choice(self.responses.get("Answer"))
        self.recommendation=random.choice(self.suggestions)
        self.suggestions.remove(self.recommendation)
        return answer.format(self.recommendation)
            
       # %%
    # -- Ivan -- look for matches with preferences in the database
    
    def lookup(self,user_preferences):
        """
        Look for restaurants in database using user preferences
    
        Parameters
        ----------
        user_preferences : list
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
        
        if user_preferences[0] == "any" or user_preferences[0] == 0:
            fit_area = set(range(len(self.area)))
        else:
            for i,a in enumerate(self.area):
                if a == user_preferences[0]:
                    fit_area.add(i)
        if user_preferences[1] == "any" or user_preferences[1] == 0:
            fit_price = set(range(len(self.price_range)))
        else:
            for j,p in enumerate(self.price_range):
                if p == user_preferences[1]:
                    fit_price.add(j)
        if user_preferences[2] == "any" or user_preferences[2] == 0:
            fit_food = set(range(len(self.food_types)))
        else:
            for k,f in enumerate(self.food_types):
                if f == user_preferences[2]:
                    fit_food.add(k)
        option_numbers = fit_area.intersection(fit_price, fit_food)
        if option_numbers:
            for i in option_numbers:
                res.append(self.restaurant_names[i])
                
        return res
    

    
    # %%
    '''
    
    statelog = list()
    res=[] #list of restaurant recommendations
    if __name__ == "__main__":
        dialogue("", "init", [0,0,0])
        '''