import scipy.cluster.hierarchy as HieraarchicalClustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
# creating data for Hierarchical Clustering
df = np.random.randint(0,10,size = [100,2])
X1 = np.random.randint(0,4,size = [300,1])
X2 = np.random.uniform(0,10,size = [300,1])
df = np.append(X1,X2,axis = 1)
hierCl = HieraarchicalClustering.linkage(df, method='ward')
# getting the dendogram to pick right number of clusters
dendrogram = HieraarchicalClustering.dendrogram(hierCl)
plt.title('Dendrogram')
plt.xlabel("Observations")
plt.ylabel('Euclidean distances')
plt.show()
Hcl= AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage ='ward')
Hcl_fitted = Hcl.fit_predict(df)
df = pd.DataFrame(df)
df["labels"] = Hcl_fitted

# Visualizing the clustered data
plt.scatter(df[df["labels"] == 0][0], df[df["labels"] == 0][1],
c='black', label='cluster 1')
plt.scatter(df[df["labels"] == 1][0], df[df["labels"] == 1][1],
c='green', label='cluster 2')
plt.scatter(df[df["labels"] == 2][0], df[df["labels"] == 2][1],
c='red', label='cluster 3')
plt.scatter(df[df["labels"] == 3][0], df[df["labels"] == 3][1],
c='magenta', label='cluster 4')
plt.scatter(df[df["labels"] ==4][0], df[df["labels"] == 4][1],
c='purple', label='cluster 5')
plt.scatter(df[df["labels"] == 5][0], df[df["labels"] == 5][1],
c='y', label='cluster 6')
plt.scatter(df[df["labels"] ==6][0], df[df["labels"] == 6][1],
c='black', label='cluster 7')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hierarchical Clustering ')
plt.show()