# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
dataset1 = pd.read_csv('training_ratings_for_kaggle_comp.csv')

users = dataset1.iloc[ : , 0].values
movies_id = dataset1.iloc[ : ,1].values
ratings = dataset1.iloc[ : ,2].values

datContent = open('movies.txt','r', encoding='ISO-8859-1')
df = datContent.readlines()      
movie_name_list=[]
for i in range(0,len(df)):
    movie_name_list.append(df[i].split('::')[1])

a=users[0]
l=[]
final_list = [];
for i in range(0,500100):
    if(a==users[i]):
        if(ratings[i]>=3):
            l.append(movies_id[i])
    else:
        a=users[i]
        final_list.append(l)        
        l=[]

from apyori import apriori
rules = apriori(final_list, min_confidence = 0.2, min_lift = 3, min_length = 3)
results = list(rules)

Result_List = []
new_file = open("new_file.txt",'w')

for j in range(0,len(results)):
    lis = list(results[j].items)
    l=[]
    for i in range(0,len(lis)):
            l.append(movie_name_list[lis[i]])
            new_file.write(movie_name_list[lis[i]])
    Result_List.append(l)
    new_file.write("\n")
    
    
    
