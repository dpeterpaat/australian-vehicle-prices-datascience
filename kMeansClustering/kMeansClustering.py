import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


data = pd.read_csv("/Users/dicsypeter/Documents/Code/dataviz/avp.csv", )

n_samples = 300
n_clusters = 2
random_state = 42
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state)



x = np.array(data['Kilometres']).reshape(-1,1)
y = np.array(data['Price']).reshape(-1,1)

xnew = x[0:100].astype(int)
ynew = y[0:100].astype(int)

X=np.array([xnew,ynew])
print(X)



# kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
# kmeans.fit(X)

# print(X[:5])
# print(np.array([0.2,0.3,0.4]))

# cluster_centers = kmeans.cluster_centers_
# cluster_labels = kmeans.labels_


# plt.scatter(X[:, 0], X[:,1], c=cluster_labels, cmap='viridis')
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', c='red', s=100, label='Cluster Centers')
# plt.title('K-Means Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()

