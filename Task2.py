import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("C:/Users/Hema M/Downloads/Mall_Customers.csv")

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

data['Cluster'] = clusters

plt.figure(figsize=(8, 6))

plt.scatter(X.iloc[clusters == 0, 0], X.iloc[clusters == 0, 1], s=100, label='Cluster 1')
plt.scatter(X.iloc[clusters == 1, 0], X.iloc[clusters == 1, 1], s=100, label='Cluster 2')
plt.scatter(X.iloc[clusters == 2, 0], X.iloc[clusters == 2, 1], s=100, label='Cluster 3')
plt.scatter(X.iloc[clusters == 3, 0], X.iloc[clusters == 3, 1], s=100, label='Cluster 4')
plt.scatter(X.iloc[clusters == 4, 0], X.iloc[clusters == 4, 1], s=100, label='Cluster 5')

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='black',
    label='Centroids'
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.show()

print(data.head())
