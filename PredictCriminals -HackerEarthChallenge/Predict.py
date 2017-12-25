
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
x_testdata = sc.transform(x_testdata)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators = 100 ,min_child_weight = 1, gamma = 1 , max_depth = 6)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

y_final_pred = classifier.predict(x_testdata)

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


# n_estimators = 300 , max_depth = 3 , min_child_weight = 3

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate':[0.1,0.2,0.3]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           return_train_score=False)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Predicting the Test set results



df = pd.DataFrame(np.array(t).reshape(1,11430), columns = list("PERID"))

test = []

for i in range(0,11430):
    print (t[i])    
    test.append({t[i] , y_final_pred[i]})

df = pd.DataFrame(np.array(test), columns = list("PERID"))    
test.to_csv('first.csv',index=False)



# Making the Confusion Matrix

# Applying k-Fold Cross Validation
accuracies.std()



