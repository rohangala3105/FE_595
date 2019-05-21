# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

# import iris data
iris_dataset = pd.read_csv('iris.csv')

iris_features = iris_dataset.iloc[:, [0,1, 2, 3]].values

no_of_clust = []

#Making n number of CLusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(iris_features)
    no_of_clust.append(kmeans.inertia_)
 
    
for item in range(len(no_of_clust)-1):
    print(abs(no_of_clust[item+1] - no_of_clust[item]))    

    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), no_of_clust)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Values within Cluster Sum of Squares') #within cluster sum of squares
plt.show()


#Applying the best kmeans to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(iris_features)

label_color = ["Orange","Pink","Navy"]
label = ['Iris-setosa', 'Iris-versicolour','Iris-virginica']
#Visualising the clusters  
for plot_no in range(3):
    plt.scatter(iris_features[y_kmeans == plot_no, 0], iris_features[y_kmeans == plot_no, 1], s = 100, c = label_color[plot_no], label = label[plot_no])

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()



