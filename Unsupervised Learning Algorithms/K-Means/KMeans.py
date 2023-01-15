from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


# ------------------------------- Function for performing K-means  -------------------------------#
def KMeans_Algorithm(df,K):
    # randomly assign each observation to a cluster
    # obtain the centroids of the clusters
    # reassign observations to clusters with the closest cerntroid
    KMeans_model = KMeans(n_clusters=K,init = 'k-means++',max_iter = 300,random_state = 2021)
    KMeans_model.fit(df)
    # storing the centroids
    centroids = KMeans_model.cluster_centers_
    centroids_df = pd.DataFrame(centroids,columns = ["X","Y"])
    # getting the labels/classes
    labels = KMeans_model.labels_
    df = pd.DataFrame(df)
    df["labels"] = labels
    return(df, centroids)

# ------------------------------ Creating data for K-Means Clustering -----------------------------#
df = np.random.randint(0,10,size = [100,2])
X1 = np.random.randint(0,4,size = [300,1])
X2 = np.random.uniform(0,10,size = [300,1])
df = np.append(X1,X2,axis = 1)


# ----------------------------- Elbow Method for optimal K---------------------------------- # 
def Elbow_Method(df):
    inertia = []
    # considering K = 1,2,...,10 as K
    K = range(1, 10)
    for k in K:
        KMeans_Model = KMeans(n_clusters=k, random_state = 2022)
        KMeans_Model.fit(df)
        inertia.append(KMeans_Model.inertia_)
    return(inertia)

K = range(1, 10)
inertia = Elbow_Method(df)
plt.figure(figsize = (8,8))
plt.plot(K, inertia, 'bx-', color = "forestgreen")
plt.xlabel("K: number of clusters")
plt.ylabel("Inertia")
plt.title("K-Means: Elbow Method")
plt.show()

# ----------------------------- Visualizing the raw data ----------------------------------- # 
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color = "forestgreen")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Visualization of raw data');
plt.show()

# ---------------------------- K-Means Clustering with Optimal K -------------------------- #
Clustered_df = KMeans_Algorithm(df = df,K =4)
df = pd.DataFrame(Clustered_df[0])
centroids = Clustered_df[1]
# ----------------------------- Visualizing the clustered data ----------------------------- # 
fig, ax = plt.subplots(figsize=(6, 6))
# for observations with each type of labels from column 1 and 2
plt.scatter(df[df["labels"] == 0][0], df[df["labels"] == 0][1],
c='black', label='cluster 1')
plt.scatter(df[df["labels"] == 1][0], df[df["labels"] == 1][1],
c='green', label='cluster 2')
plt.scatter(df[df["labels"] == 2][0], df[df["labels"] == 2][1],
c='red', label='cluster 3')
plt.scatter(df[df["labels"] == 3][0], df[df["labels"] == 3][1],
c='y', label='cluster 4')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='black', label='centroid')
plt.legend()
plt.xlim([-2, 6])
plt.ylim([0, 10])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means: Visualization of clustered data \n K = 4')
ax.set_aspect('equal')
plt.show()


