import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('matches.csv')

# 4 - team1 
# 5 - team2
# 6 - toss_winner
# 7 - toss_decesion
# 8 -result
# 14 - venue

# 10 - Winner 

X = dataset.iloc[ : , [4,5,6,7,8,14]].values
Y = dataset.iloc[ : , [10]].values 

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
labelencoder_x1 = LabelEncoder()
X[:,1] = labelencoder_x1.fit_transform(X[:,1])
labelencoder_x2 = LabelEncoder()
X[:,2] = labelencoder_x2.fit_transform(X[:,2])
labelencoder_x3 = LabelEncoder()
X[:,3] = labelencoder_x3.fit_transform(X[:,3])
labelencoder_x4 = LabelEncoder()
X[:,4] = labelencoder_x4.fit_transform(X[:,4])
labelencoder_x4 = LabelEncoder()
X[:,5] = labelencoder_x4.fit_transform(X[:,5])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

y_s = labelencoder_x.classes_

import array

y_s1 = array.array('i')

count=0

for i,j in enumerate(Y):
    if(pd.isnull(j)):
        y_s1.append(14)
        continue
    for i1,j1 in enumerate(y_s):  
      if(j == j1):
          count+=1
          print (str(i) + " " + str(i1))
          y_s1.append(i1)

print("count " + str(count))

len(y_s1)                  
len(Y)

Y_set = np.array(y_s1)

Y = Y_set

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X , Y ,test_size = 0.15 , random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


########################################################
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf' ,random_state = 0)
classifier.fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy' ,random_state = 0)
classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)

count = 0
for i,j in enumerate(y_pred):
    if(y_test[i] == j):
        count+=1
    
print(count)

print(52/96)

################################################################

# Importing the Keras libraries and packages

from keras.utils import to_categorical
y_train = to_categorical(y_train)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 82))

# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, epochs = 300)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = np.argmax(y_pred , axis=1)

