
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('criminal_train.csv')
testing_data = pd.read_csv('criminal_test.csv')
x_testdata = testing_data.iloc[:, 1:71].values
X = dataset.iloc[:, 1:71].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
x_testdata = sc.transform(x_testdata)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 36, kernel_initializer = 'uniform', activation = 'relu', input_dim = 70))

# Adding the second hidden layer
classifier.add(Dense(units = 36, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 36, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 36, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

y_final_pred = classifier.predict(x_testdata)
y_final_pred = (y_final_pred > 0.5)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
y_final_pred = labelencoder_X_1.fit_transform(y_final_pred)


t = testing_data['PERID']

test = []
yp = []
for i in range(0,11430):
    test.append(t[i])
    yp.append(y_final_pred[i])

from collections import OrderedDict

df = OrderedDict([('PERID' , test),('Criminal' , yp)])
df = pd.DataFrame.from_dict(df)    
df.to_csv('first.csv',index=False)

