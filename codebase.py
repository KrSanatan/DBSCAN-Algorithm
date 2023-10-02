import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics

# Taking input 
clusterBlobData=pd.read_csv("C:\\Users\\HP\\Desktop\\IIT_Patna_MTech_AI\\sem_1\\ML\\assignment2\\cluster_blobs.csv")
clusterCirclesData=pd.read_csv("C:\\Users\\HP\\Desktop\\IIT_Patna_MTech_AI\\sem_1\\ML\\assignment2\\cluster_circles.csv")
clusterMoonsData=pd.read_csv("C:\\Users\\HP\\Desktop\\IIT_Patna_MTech_AI\\sem_1\\ML\\assignment2\\cluster_moons.csv")

# Data cleaning
clusterBlobData.fillna(method ='ffill', inplace = True)
clusterCirclesData.fillna(method='ffill', inplace = True)
clusterMoonsData.fillna(method='ffill', inplace = True)

# Cluster blobs using DBSCAN
mainClusterBlobDataDbscan = pd.DataFrame(clusterBlobData)  
dbscan= DBSCAN(eps = 1.5, min_samples = 3).fit(mainClusterBlobDataDbscan).labels_
plt.figure(figsize =(10, 10))
mainClusterBlobDataDbscan.columns=['X1','X2']
plt.scatter(mainClusterBlobDataDbscan['X1'], mainClusterBlobDataDbscan['X2'],c=dbscan, cmap ='rainbow') 
plt.title("DBSCAN")
plt.show()
print("The Silhoutte score of the dataset cluster_blobs using DBScan is => ",
metrics.silhouette_score(mainClusterBlobDataDbscan,dbscan))

# Cluster blobs using Kmeans
mainClusterBlobDataKmeans = pd.DataFrame(clusterBlobData)  
kmeans= KMeans(3).fit(mainClusterBlobDataKmeans).labels_
plt.figure(figsize =(10, 10))
mainClusterBlobDataKmeans.columns=['X1','X2']
plt.scatter(mainClusterBlobDataKmeans['X1'], mainClusterBlobDataKmeans['X2'],c=kmeans, cmap ='rainbow')
plt.title("Kmeans")
plt.show()
print("The Silhoutte score of the dataset cluster_blobs using Kmeans is => ",
metrics.silhouette_score(mainClusterBlobDataKmeans,kmeans))

# Cluster circles using DBSCAN
mainClusterCirclesDataDbscan = pd.DataFrame(clusterCirclesData)  
dbscan= DBSCAN(eps = 0.15, min_samples = 3).fit(mainClusterCirclesDataDbscan).labels_
plt.figure(figsize =(10, 10))
mainClusterCirclesDataDbscan.columns=['X1','X2']
plt.scatter(mainClusterCirclesDataDbscan['X1'], mainClusterCirclesDataDbscan['X2'],c=dbscan, cmap ='rainbow')
plt.title("DBSCAN")
plt.show()
print("The Silhoutte score of the dataset cluster_circles using DBScan => ",
metrics.silhouette_score(mainClusterCirclesDataDbscan,dbscan))

# Cluster circles using Kmeans
mainClusterCirclesDataKmeans = pd.DataFrame(clusterCirclesData)  
kmeans_1= KMeans(2).fit(mainClusterCirclesDataKmeans).labels_
plt.figure(figsize =(10, 10))
mainClusterCirclesDataKmeans.columns=['X1','X2']
plt.scatter(mainClusterCirclesDataKmeans['X1'], mainClusterCirclesDataKmeans['X2'],c=kmeans_1, cmap ='rainbow')
plt.title("Kmeans")
plt.show()
print("The Silhoutte score of the dataset cluster_circles using Kmeans is => ",
metrics.silhouette_score(mainClusterCirclesDataKmeans,kmeans_1))

# Cluster moons using DBSCAN
mainClusterMoonsDataDbscan = pd.DataFrame(clusterMoonsData)  
dbscan= DBSCAN(eps = 0.15, min_samples = 3).fit(mainClusterMoonsDataDbscan).labels_
plt.figure(figsize =(10, 10))
mainClusterMoonsDataDbscan.columns=['X1','X2']
plt.scatter(mainClusterMoonsDataDbscan['X1'], mainClusterMoonsDataDbscan['X2'],c=dbscan, cmap ='rainbow') 
plt.title("DBSCAN")
plt.show()
print("The Silhoutte score of the dataset cluster_moons using DBScan is => ",
metrics.silhouette_score(mainClusterMoonsDataDbscan,dbscan))

# Cluster moons using Kmeans
mainClusterMoonsDataKmeans = pd.DataFrame(clusterMoonsData)  
kmeans_2= KMeans(2).fit(mainClusterMoonsDataKmeans).labels_
plt.figure(figsize =(10, 10))
mainClusterMoonsDataKmeans.columns=['X1','X2']
plt.scatter(mainClusterMoonsDataKmeans['X1'], mainClusterMoonsDataKmeans['X2'],c=kmeans_2, cmap ='rainbow')
plt.title("Kmeans")
plt.show()
print("The Silhoutte score of the dataset cluster_moons using Kmeans is => ",
metrics.silhouette_score(mainClusterMoonsDataKmeans,kmeans_2))