# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 18:11:48 2025

@author: mahes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\mahes\OneDrive\Documents\Social_Network knn.csv")
dataset

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#spliting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#training the knn model on the trainingset
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=4,p=1)
classifier.fit(x_train,y_train)

#predicting the test set results
y_pred=classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#model acuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

#classification report
from sklearn.metrics import classification_report
cr=classifier.score(x_train,y_train)
cr

bias=classifier.score(x_train,y_train)
bias

variance=classifier.score(x_test,y_test)
variance
