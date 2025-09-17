import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\mahes\OneDrive\Documents\Churn xgboost.csv")
dataset

x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values
print(x)
print(y)

#encoding categorical data
#label encoding the gender column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
print(x)

#inehot encoding the "geography " column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)


#splitting dataset into training 
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2,random_state=0)

#training xg boost on the training set
from xgboost import XGBClassifier
classifier=XGBClassifier(random_state=0)
classifier.fit(x_train,y_train)

#prediting test results
y_pred=classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

bias=classifier.score(x_train,y_train)
print(bias)

variance=classifier.score(x_test,y_test)
print(variance)
