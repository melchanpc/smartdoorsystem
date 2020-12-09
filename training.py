# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:48:33 2020

@author: MelanieChan
"""

# compare algorithms
# Everything should load without error. If you have an error, stop. You need a working SciPy
# environment before continuing.
import os
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import joblib
#Load dataset
#If you do have network problems, you can download the iris.csv file into your working directory
#and load it using the same method, changing URL to the local file name.


data= read_csv('data.csv')
dataset = DataFrame(data,columns=['distance1','distance2','distance3','relativediff1','relativediff2','class','Absdiff1','Absdiff2','speed'])
# #We can get a quick idea of how many instances (rows) and how many attributes (columns) the
# 
# #contains with the shape property. You should see 150 instances and 5 attributes
# #shape
# print(dataset.shape)
# #You should see the first 20 rows of the data:
# # head
# print(dataset.head(20))
# #Now we can take a look at a summary of each attribute.
# #This includes the count, mean, the min and max values as well as some percentiles.
# #descriptions
# #We can see that all of the numerical values have the same scale (centimeters)
# #and similar ranges between 0 and 8 centimeters.
# print(dataset.describe())
# #Let’s now take a look at the number of instances (rows) that belong to each class.
# #We can view this as an absolute count.
# #We can see that each class has the same number of instances (50 or 33% of the dataset).
# #class distribution
# print(dataset.groupby('class').size())
# #Given that the input variables are numeric, we can create box and whisker plots of each.
# #This gives us a much clearer idea of the distribution of the input attributes
# #box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False)
# pyplot.show()
# #We can also create a histogram of each input variable to get an idea of the distribution.
# #It looks like perhaps two of the input variables have a Gaussian distribution.
# #This is useful to note as we can use algorithms that can exploit this assumption.
# #histograms
# dataset.hist()
# pyplot.show()
# #First, let’s look at scatterplots of all pairs of attributes.
# #This can be helpful to spot structured relationships between input variables.
# #Note the diagonal grouping of some pairs of attributes.
# #This suggests a high correlation and a predictable relationship.
# #scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()
#We will split the loaded dataset into two, 80% of which we will use to train,
#evaluate and select among our models,
#and 20% that we will hold back as a validation dataset.
#You now have training data in the X_train and Y_train for preparing models and
#a X_validation and Y_validation sets that we can use later.
#Notice that we used a python slice to select the columns in the NumPy array.
#Split-out validation dataset
array = dataset.values
X = array[:,3:5]
y = array[:,5]
X2 = array[:,6:8]
y2 = array[:,8]
print(dataset.groupby('class').size())
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1,
shuffle=True)
#Let’s test 3 different algorithms:
#K-Nearest Neighbors (KNN).

#Classification and Regression Trees (CART).
#Support Vector Machines (SVM).
#This is a good mixture of nonlinear (KNN, CART and SVM) algorithms.
# Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))
#We now have 3 models and accuracy estimations for each.
#We need to compare the models to each other and select the most accurate.
#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=12, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Make predictions on validation dataset
model = DecisionTreeClassifier()

model.fit(X_train, Y_train)
filename = 'finalized_model.sav'
joblib.dump(model, filename)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#training speed


print(dataset.groupby('speed').size())
X_train, X_validation, Y_train, Y_validation = train_test_split(X2, y2, test_size=0.20, random_state=1,
shuffle=True)
#Let’s test 3 different algorithms:
#K-Nearest Neighbors (KNN).

#Classification and Regression Trees (CART).
#Support Vector Machines (SVM).
#This is a good mixture of nonlinear (KNN, CART and SVM) algorithms.
# Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))
#We now have 3 models and accuracy estimations for each.
#We need to compare the models to each other and select the most accurate.
#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=12, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Make predictions on validation dataset
model = DecisionTreeClassifier()

model.fit(X_train, Y_train)
filename = 'finalized_model2.sav'
joblib.dump(model, filename)
predictions = model.predict(X_validation)

#We can evaluate the predictions by comparing them to the expected results
#in the validation set, then calculate classification accuracy,
#as well as a confusion matrix and a classification report.
#Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#We can see that the accuracy is 0.966 or about 96% on the hold out dataset.
#The confusion matrix provides an indication of the three errors made.
#Finally, the classification report provides a breakdown of each class by precision,
#recall, f1-score and support showing excellent results
#(granted the validation dataset was small).