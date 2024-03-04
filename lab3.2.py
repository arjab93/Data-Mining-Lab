import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
data = np.random.rand(10000, 2) * 100
mbk = MiniBatchKMeans(n_clusters=5, init="random", batch_size=500)
t0 = time.time()
mbk.fit(data)
t1 = time.time()
tt = t1 - t0
print("Total Time: ", tt)
centers = mbk.cluster_centers_
labels = mbk.labels_
print("Cluster Centers:", centers)
# print("Labels:", labels)
colors = ["g", "r", "b", "y", "m"]
markers = ["+", "x", "*", ".", "d"]
for i in range(len(data)):
 plt.plot(data[i][0], data[i][1], color=colors[labels[i]], 
marker=markers[labels[i]])
plt.scatter(centers[:, 0], centers[:, 1], marker="o", s=50, 
linewidths=5)
plt.show()
