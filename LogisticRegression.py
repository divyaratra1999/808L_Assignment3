import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy import genfromtxt

# Loading the Iris dataset from Sklearn
print("Iris Dataset -> ")
iris = load_iris()
x = iris.data
y = iris.target

# Splitting the dataset for training and testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)

# Creating a logistic regression model for Sklearn Iris Dataset
iris_model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
iris_model.fit(x_train, y_train)

# Making predictions on the test data
y_pred = iris_model.predict(x_test)
print("Predicted value for the Iris dataset: ",y_pred)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#---------------------------------------
print("Wiki DataSet -> ")

wikiDataset = genfromtxt('wikiData.csv', delimiter=',',dtype=None,encoding=None)
x = pd.DataFrame(wikiDataset[1:,:1])
y = pd.DataFrame(wikiDataset[1:,1])
y = y.values.ravel()

# Splitting the dataset for training and testing 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=12)

#Creating a Logistic Regression model for the Wiki DataSet
wikiModel=linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
wikiModel.fit(x_train,y_train)

# Making predictions on the test data
y_pred=wikiModel.predict(x_test)
print("Predicted value for the Wiki dataset: ",y_pred)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#---------------------------------------
print("Partial Iris DataSet -> ")

irisDataset = genfromtxt('IrisNew.csv', delimiter=',',dtype=None,encoding=None)
# Extracting the features that is the first four columns
x = pd.DataFrame(irisDataset[1:,:4])
# Extracting the target variable that is the 5th column
y = pd.DataFrame(irisDataset[1:,4])
y = y.values.ravel()

# Splitting the dataset for training and testing 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=12)

#Creating a Logistic Regression model for the Wiki DataSet
partial_iris_model=linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
partial_iris_model.fit(x_train,y_train)

# Making predictions on the test data
y_pred=partial_iris_model.predict(x_test)
print("Predicted value for the Wiki dataset: ",y_pred)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)