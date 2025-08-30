# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 19:09:02 2025

@author: mahes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\mahes\OneDrive\Documents\Data.csv")
df

#seperate x and y(indepedent variable and depedent variable)

x=df.iloc[:,:-1]
y=df.iloc[:,3]

from sklearn.impute import SimpleImputer
#create the imputer-here we replace missing values with the mean median mode

imputer=SimpleImputer()
imputer
#fit the imputer on numeric columns(age and salary)
imputer.fit(x[["Age","Salary"]])

#transform and replace the values
x[["Age","Salary"]]=imputer.transform(x[["Age","Salary"]])
x
y
from sklearn.preprocessing import LabelEncoder

#create encoder object
le=LabelEncoder()
 #Encode the state column in x x['State']=le.fit_transform(x["State"])
#encode y(purchased column)
y=le.fit_transform(y)
print(x)
print(y)
