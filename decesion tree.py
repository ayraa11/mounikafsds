import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"C:\Users\mahes\OneDrive\Documents\logit classification.csv")
dataset

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.20,random_state=0)

#standarization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)



from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini',max_depth=10)
classifier.fit(x_train,y_train)

''''''
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(max_depth=4,n_estimators=30,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
''''''
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

bias=classifier.score(x_train,y_train)
bias

variance=classifier.score(x_test,y_test)
variance

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)
