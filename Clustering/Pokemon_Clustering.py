import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('pokemon.csv')
X = dataset.iloc[ : , [2,4]].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])

from sklearn.cluster import KMeans 
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:"), fig_size
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'orange', label = 'Cluster 5')
plt.xlabel('Type of Pokemon')
plt.ylabel('HP of Pokemon')
plt.legend()
plt.show()

def takeSecond(elem):
    return elem[0]

labels = kmeans.labels_
names = dataset.iloc[ : ,[1]].values
final_list=[]
for i,j in enumerate(names):
    l=[]
    l.append(str(labels[i]))
    l.append(str(names[i]))
    final_list.append(l)
    
final_list.sort(key = takeSecond) 

f = open("PokemonResult", "w")
for i in final_list:
    print(i)
    f.write(i[0] + " -> " + i[1] + '\n')    