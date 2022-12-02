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
data = pd.read_csv("Data.csv")
 
# 4. Split data into training and test sets
array = data.values
X = array[1:,0:11]
y = array[1:,11]
# print(X)
# print("y",y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
#5 Analyse the dataset

print(data.head())

i=0
for i in range(len(data))
 boxplot.data(x[i])
'''
The above loop gets us the boxplots of all the variables
'''


import matplotlib.pyplot as plt
print("--------------------------------------")

#plotting the results
# plt.scatter(X_test[:,0], pred[:])
# plt.show()
