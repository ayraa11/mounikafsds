# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 11:57:29 2025

@author: mahes
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\mahes\OneDrive\Documents\emp_sall.csv")
dataset

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#svm model
from sklearn.svm import SVR
svr_regressor=SVR(kernel='poly',degree=4,gamma='scale',C=10.0)
svr_regressor.fit(x,y)

svr_model_pred=svr_regressor.predict([[6.5]])
print(svr_model_pred)

#knn model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=5,weights='distance')
knn_reg_model.fit(x,y)

knn_reg_pred=knn_reg_model.predict([[6.5]])
print(knn_reg_pred)
