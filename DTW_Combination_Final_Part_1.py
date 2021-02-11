#
from dtaidistance import dtw_visualisation as dtwvis
import math
from sklearn.cluster import KMeans
import xlsxwriter
import os
from itertools import combinations
from dtaidistance import dtw
from numpy.core._multiarray_umath import ndarray
from sklearn import cluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import sympy
# Retrieve current working directory

cwd = os.getcwd()
cwd

# List directory
os.listdir('.')

# Import the data
df = pd.read_excel(r'C:\Users\EA\Desktop\Paper\data.xlsx')
#print(df)

timeseries = np.array([
    pd.Series(df['G1'] + 0.307808),
    pd.Series(df['G2'] - 68.187345),
    pd.Series(df['G3'] - 59.33976)])

#timeseries =np.array ([[1, 2, 4, 7, 8, 9],[0, 3, 5, 6, 7, 8],[1, 2, 3, 4, 7, 6]])

def rSubset(arr, r):
    return list(combinations(arr, r))

# Function
if __name__ == "__main__":
    arr = timeseries
    r = 2

comb_matrix = (rSubset(arr, r))
#print(comb_matrix)

comb_matrix_2 = np.array(comb_matrix)
#print(comb_matrix_2)

final_matrix = []
distances = []
for m in range(comb_matrix_2.shape[0]):
    d, paths = dtw.warping_paths(comb_matrix_2[m,0],comb_matrix_2[m,1])
    best_path = dtw.best_path(paths)
    #dtwvis.plot_warpingpaths(comb_matrix_2[m,0], comb_matrix_2[m,1], paths, best_path)
    new_matrix = np.diag(paths)
    row = []
    distances.append(d)
    row.append(paths)
    final_matrix.append(new_matrix)
    #final_matrix=np.append(final_matrix,row)
    #final_matrix.append(paths)
    #plt.plot(final_matrix)
    #print(final_matrix)
    #print(paths)
    #plt.plot(final_matrix)
    #plt.show()

print(final_matrix)
print('++++++++++++++++++++++++++++++++++++++++++++')
X = np.array(final_matrix)
#kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
#pred_y = kmeans.fit_predict(X)
#plt.scatter(X[:,0], X[:,1])
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
#plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=300, c='red')
#plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=300, c='red')
#plt.show()

dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
#4 Fitting hierarchical clustering to the Mall_Customes dataset
# There are two algorithms for hierarchical clustering: #Agglomerative Hierarchical Clustering and
# Divisive Hierarchical Clustering. We choose Euclidean distance and ward method for our algorithm class
# from sklearn.cluster import AgglomerativeClustering
# hc = AgglomerativeClustering(n_clusters = 2, affinity='precomputed', linkage='complete')
# # Lets try to fit the hierarchical clustering algorithm  to dataset #X while creating the clusters vector that tells for each customer #which cluster the customer belongs to.
# y_hc=hc.fit_predict(X)
# #5 Visualizing the clusters. This code is similar to k-means #visualization code. We only replace the y_kmeans vector name to #y_hc for the hierarchical clustering
# plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=50, c='red', label ='Cluster 1')
# plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=50, c='blue', label ='Cluster 2')
# #plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=50, c='green', label ='Cluster 3')
# plt.title('Generator coherence plot)')
# plt.xlabel('Time')
# plt.ylabel('Phase')
# plt.show()

df = pd.DataFrame(X).T
df.to_excel(excel_writer = "C:/Users/EA/Desktop/Paper/results.xlsx")