# !pip install scikit-learn-extra
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
import matplotlib.pyplot as plt
iris_data = load_iris()
x = iris_data.data
y = iris_data.target
# print(x[:5])
# print(y[:5])
sc = StandardScaler().fit(x)
sx = sc.transform(x)
km = KMedoids(n_clusters=3)
km.fit(sx)
py = km.fit_predict(sx)
# print("Predicted: ", py)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
colors = ["g", "r", "b"]
markers = ["+", "x", "*"]
for i in range(len(sx)):
 ax.scatter(sx[i][0], sx[i][1], sx[i][2], color=colors[py[i]], 
marker=markers[py[i]])
plt.show()
ri = metrics.rand_score(y, py)
print("Rand Index:", ri)
hs = metrics.homogeneity_score(y, py)
print("Homogeniety Score:", hs)
cs = metrics.completeness_score(y, py)
print("Completeness Score:", cs)
sc = metrics.silhouette_score(sx, py, metric="euclidean")
print("Silhouette Coefficient:", sc)