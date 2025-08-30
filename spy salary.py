# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 12:27:43 2025

@author: mahes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\mahes\Downloads\Salary_Data.csv")
dataset

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

m=regressor.coef_

c=regressor.intercept_
(m*12)+c
(m*12)+c

dataset.mean()
dataset['Salary'].mean()
dataset.median()
dataset['Salary'].median()
dataset.mode()
dataset['Salary'].mode()
dataset.std()
dataset['Salary'].std()


#coefficient variation

from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])

#corelation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

#skewness
dataset.skew()
dataset['Salary'].skew()

#standed error
dataset.sem()
dataset['Salary'].sem()

#z score
import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

#degrees of freedom
a=dataset.shape[0]
b=dataset.shape[1]

degree_of_freedom=a-b
print(degree_of_freedom)

#ssr
y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)

print(SSR)
#SST
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

#SSE
y=y[0:6]
SSE=np.sum((y_pred)**2)

print(SSE)

#R2
r_square=1-SSR/SST
print(r_square)





