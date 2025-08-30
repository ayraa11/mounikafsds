# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:52:11 2025

@author: mahes
"""
#svr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\mahes\OneDrive\Documents\emp_sall.csv")
dataset
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#fitting svr into the dataset
from sklearn.svm import SVR

#IMPORTED THE SVR CLASS FROM SKLEARN.SVM
regressor=SVR(kernel='poly',degree=5)

#create regressor object and for now understand kernel is use for linear.polyor non linear svr  
#non linear data we will use kernel and rbf
regressor.fit(x,y)

#predicting new result
y_pred=regressor.predict([[6.5]])

#visualization the svr result
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('postion level')
plt.ylabel('salary')
plt.show()
