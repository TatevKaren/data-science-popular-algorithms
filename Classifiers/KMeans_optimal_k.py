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

