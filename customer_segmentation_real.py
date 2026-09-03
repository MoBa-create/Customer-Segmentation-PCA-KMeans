from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
df_raw = pd.read_csv(url)

print("=== First 5 rows in (Mall Customers) ===")
print(df_raw.head())

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
x = df_raw[features]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

explained_variance = pca.explained_variance_ratio_
print(f"\n=== Percentage of variance retained (PCA Variance) ===")
print(f"PC1 : {explained_variance[0]*100} %")
print(f"PC2 : {explained_variance[1]*100} %")
print(f"Total variance retained : {sum(explained_variance)*100} %")

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--", color="b")
plt.title("Elbow Method for Real Mall Customer Data")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig(os.path.join(OUTPUTS_DIR, "elbow_method.png"))
plt.close()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
df_raw["Cluster"] = kmeans.fit_predict(x_pca)

pca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
pca_df["Cluster"] = df_raw["Cluster"]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x="PC1", y="PC2", hue="Cluster", data=pca_df,
    palette="Set1", s=100, style="Cluster", alpha=0.9
)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c="yellow", marker="X", label="Centroids")

plt.title(f'Real Customer Segments (PCA + K-Means)\nVariance Retained: {sum(explained_variance)*100} %')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100} %)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100} %)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUTS_DIR, "customer_clusters_pca.png"))
plt.show()

print("\n=== Average characteristics per real customer segment ===")
profile = df_raw.groupby('Cluster')[features].mean()
print(profile)

joblib.dump(scaler, os.path.join(OUTPUTS_DIR, 'scaler.pkl'))
joblib.dump(pca, os.path.join(OUTPUTS_DIR, 'pca_transformer.pkl'))
joblib.dump(kmeans, os.path.join(OUTPUTS_DIR, 'kmeans_model.pkl'))
            
print("\n === Updated with real data and saved results in a folder === 'outputs'!")