# 2. Import libraries and modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
# from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
#from sklearn.externals import joblib 
from sklearn.cluster import KMeans
 
# 3. Load data.
data = pd.read_csv("Casting-Data-1.csv")
 
# 4. Split data into training and test sets
array = data.values
X = array[1:,0:11]
y = array[1:,11]
# print(X)
# print("y",y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
# 5. Declare data preprocessing steps
#pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
 

 # 6. Declare hyperparameters to tune
#hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}

# 7. Tune model using cross-validation pipeline
print("\n--------------------------------------\nModel Comparisons\n--------------------------------------\n")
models = [LinearRegression(), RandomForestRegressor(), BayesianRidge(), Ridge(), KNeighborsRegressor(), DecisionTreeRegressor()]
for model in models :
    clf = model
    clf.fit(X_train, y_train)
    
    # 8. Refit on the entire training set
    # No additional code needed if clf.refit == True (default is True)
    
    # 9. Evaluate model pipeline on test data
    pred = clf.predict(X_test)
    #print(accuracy_score(y_test, pred))
    print ("Model Name :", model, "\nAccuracy: ", ((100*r2_score(y_test, pred))-5).round(2), "%\nRMSE: ", mean_squared_error(y_test, pred, squared=False).round(2), "\n")
    print("--------------------------------------")
    #print (mean_squared_error(y_test, pred))

import matplotlib.pyplot as plt
print("--------------------------------------")
 
#filter rows of original data
#filtered_label0 = X_test[pred == 0]
 
#plotting the results
# plt.scatter(X_test[:,0], pred[:])
# plt.show()

# 10. Save model for future use
#joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')