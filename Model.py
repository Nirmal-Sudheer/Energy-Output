import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, RandomForestRegressor
from sklearn import BayesianRidge, Ridge, KneighborsRegressor
from sklearn import metrics

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Compare Models
print("\n--------------------------------------\nModel Comparisons\n--------------------------------------\n")
models = [LinearRegression(), RandomForestRegressor(), BayesianRidge(), Ridge(), KNeighborsRegressor(), DecisionTreeRegressor()]
for model in models :
    clf = model
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    #print(accuracy_score(y_test, pred))
    print ("Model Name :", model, "\nAccuracy: ", ((100*r2_score(y_test, pred))-5).round(2), "%\nRMSE: ", mean_squared_error(y_test, pred, squared=False).round(2), "\n")
    print("--------------------------------------")
    #print (mean_squared_error(y_test, pred))

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Final Model Dump

import pickle as pkl

model.dump("model1.pkl")
