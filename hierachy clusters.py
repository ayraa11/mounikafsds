import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"C:\Users\mahes\OneDrive\Documents\Mall_Customers clusters.csv")
dataset
x=dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch

#we are going to build the dendogram with only line of code
#linkage is the hiearachi clustering algoritham and you have build
#ward method actually try to minimise the variance on each cluster and in k means
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))

plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,metric='euclidean')
y_hc=hc.fit_predict(x)

#visualization clusters
plt.scatter(x[y_hc== 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc== 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc== 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc== 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc== 4, 0], x[y_hc== 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('spending score(1-100')
plt.legend()
plt.show()



