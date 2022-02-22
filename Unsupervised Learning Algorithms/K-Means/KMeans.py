from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import pandas as pd
# randomly assign each observation to a cluster
# obtain the centroids of the clusters
# reassign observations to clusters with the closest cerntroid
N = 300
X1 = np.random.randint(0,4,size = [N,1])
X2 = np.random.uniform(0,10,size = [N,1])
df = pd.DataFrame(np.append(X1,X2,axis = 1))

train = df.sample(int(N*0.8))
not_train = df.drop(index = train.index)
valid = not_train.sample(int(len(not_train)/2))
test = not_train.drop(index = valid.index)
# Validation
KMean_inertia = []
for k in range(1,11):
    classifier = KMeans(n_clusters=k, max_iter=300, random_state=2021, init='k-means++')
    classifier.fit(np.array(train))
    classifier.predict(valid)
    valid_inertia = classifier.inertia_
    KMean_inertia.append(valid_inertia)
#
# import matplotlib.pyplot as plt
# k_values = np.arange(1,11,1)
# plt.plot(k_values, KMean_inertia, color = 'purple',linewidth = 2)
# plt.title("Elbow method for finding the optimal K for KMeans")
# plt.xlabel("K")
# plt.ylabel("Inertia_")
# plt.show()


# Final process Train and Predictions
trained_KMeans = KMeans(n_clusters = 3, max_iter = 300, random_state = 2021, init = 'k-means++')
trained_KMeans.fit(train)
predict = trained_KMeans.predict(test).reshape((len(test),1))
predicted_data = pd.DataFrame(np.append(test,predict,axis = 1))
predicted_data.columns = ["feature1", "feature2","cluster"]
print(predicted_data)



#labels = np.array(classifier.labels_).reshape((len(train),1))

#clustered_df = pd.DataFrame(np.append(df,labels,axis = 1))
#clustered_df.columns = ["feature_1","feature_2","cluster"]




#
#
#
# def KMeans_Algorithm(df,K):
#     KMeans_model = KMeans(n_clusters=K,init = 'k-means++',max_iter = 300,random_state = 2021)
#     KMeans_model.fit(df)
#     # storing the centroids
#     centroids = KMeans_model.cluster_centers_
#     centroids_df = pd.DataFrame(centroids,columns = ["X","Y"])
#
#     # getting the labels/classes
#     labels = KMeans_model.labels_
#     df = pd.DataFrame(df)
#     df["labels"] = labels
#     return(df)
#
# X1 = np.random.randint(0,4,size = [300,1])
# X2 = np.random.uniform(0,10,size = [300,1])
# df = np.append(X1,X2,axis = 1)
# Clustered_df = KMeans_Algorithm(df = df, K = 4)
# df = pd.DataFrame(Clustered_df)
#
# # Visualizing the raw data
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6, 6))
# plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Visualization of raw data');
# plt.show()
#
# # Plot the clustered data
# fig, ax = plt.subplots(figsize=(6, 6))
# # for observations with each type of labels from column 1 and 2
# plt.scatter(df[df["labels"] == 0][0], df[df["labels"] == 0][1],
#             c='black', label='cluster 1')
# plt.scatter(df[df["labels"] == 1][0], df[df["labels"] == 1][1],
#             c='blue', label='cluster 2')
# plt.scatter(df[df["labels"] == 2][0], df[df["labels"] == 2][1],
#             c='red', label='cluster 3')
# plt.scatter(df[df["labels"] == 3][0], df[df["labels"] == 3][1],
#             c='y', label='cluster 4')
# #plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='r', label='centroid')
# plt.legend()
# plt.xlim([-2, 6])
# plt.ylim([0, 10])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Visualization of clustered data', fontweight='bold')
# ax.set_aspect('equal')
# plt.show()
#
# #
# #
# #
# #
# #
# # def Bootstrapping(df,B):
# #     Bootrap_errors = []
# #
# #     data_orig = df
# #     num_obs_per_fold = int(len(df)/K)
# #     folds = {}
# #     for k in range(K):
# #         fold_k = df.sample(num_obs_per_fold)
# #         folds[k] = fold_k
# #         df = df.drop(index = fold_k.index)
# #
# #     for k in range(K):
# #         X_test = folds[k]["X"]
# #         Y_test = folds[k]["Y"]
# #         X_train = data_orig.drop(index = X_test.index)["X"]
# #         Y_train = data_orig.drop(index = Y_test.index)["Y"]
# #
# #         ols_model = sm.OLS(Y_train,X_train)
# #         model_fitted = ols_model.fit()
# #         Y_pred = model_fitted.predict(X_test)
# #
# #         N = len(Y_pred)
# #         MSE_test = np.sum(np.square(Y_pred-Y_test))/(N-1)
# #         CV_errors_hold_out.append(MSE_test)
# #
# #     return(CV_errors_hold_out)
# #
