# 🛍️ Advanced Customer Segmentation via PCA & K-Means

An end-to-end Machine Learning project applying Dimensionality Reduction (PCA) and Unsupervised Clustering (K-Means) on real-world retail customer data. This pipeline transforms multi-dimensional behavioral metrics into actionable customer segments for targeted business strategies.

---

## 📌 Project Architecture & Workflow

```text
.
├── customer_segmentation.py   # Complete Execution Script
├── requirements.txt            # Project Dependencies
├── .gitignore                  # Version Control Ignore Rules
├── README.md                   # Project Documentation
└── outputs/                    # Generated Artefacts & Models
    ├── customer_clusters_pca.png
    ├── elbow_method.png
    ├── scaler.pkl
    ├── pca_transformer.pkl
    └── kmeans_model.pkl

💡 Key Highlights & Performance
	Dataset Source: Real-world Mall Customers Dataset (Income, Spending Score, Age).

	Variance Retained: Reduced dimensions to 2 Principal Components while retaining 77.57% of the original variance (PC1: 44.27%, PC2: 33.31%).

	Optimal Clustering: Applied the Elbow Method to confirm 5 distinct customer groups (K = 5).

	Business Actionability: Mapped mathematical cluster profiles directly into marketing personas.


📊 Business Insights & Customer Profiling

Cluster 0 — Senior Conservative

Metrics: Avg Age: ~58 | Annual Income: ~$41.8k | Spending Score: ~35.6

Strategic Action: Promote essential products & value deals.

Cluster 1 — VIP / High Earners

Metrics: Avg Age: ~30 | Annual Income: ~$80.6k | Spending Score: ~74.0

Strategic Action: Premium loyalty programs & luxury offers.

Cluster 2 — Young Impulse Spenders

Metrics: Avg Age: ~25 | Annual Income: ~$27.8k | Spending Score: ~73.5

Strategic Action: Trend-based deals, discounts, and fashion campaigns.

Cluster 3 — Frugal High Earners

Metrics: Avg Age: ~43 | Annual Income: ~$89.7k | Spending Score: ~17.3

Strategic Action: Quality-driven marketing to encourage spending.

Cluster 4 — Moderate / Middle Class

Metrics: Avg Age: ~39 | Annual Income: ~$53.7k | Spending Score: ~44.5

Strategic Action: Standard engagement & general promotional updates.

🛠️ Tech Stack & Libraries
Language: Python 3.x

Data Engineering: Pandas, NumPy

Machine Learning: Scikit-Learn (StandardScaler, PCA, KMeans)

Visualization: Matplotlib, Seaborn

Model Serialization: Joblib

🚀 Getting Started
1. Clone the repository


git clone https://github.com/MoBa-create/customer-segmentation-pca.git
cd customer-segmentation-pca


2. Install dependencies
	pip install -r requirements.txt

3. Execute the pipeline
	python customer_segmentation.py