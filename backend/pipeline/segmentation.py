import os
import tempfile
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Force a headless, writable Matplotlib setup for CLI environments.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_segmentation(
    features_path: str = "data/processed/rfm_features.csv",
    output_path: str = "data/processed/segmented_customers.csv",
    plots_dir: str = "data/plots"
):
    """
    Runs clustering algorithms, evaluates performance, assigns business labels, and saves results.
    """
    logger.info("Starting Customer Segmentation pipeline.")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"RFM feature file not found at: {features_path}")
    
    os.makedirs(plots_dir, exist_ok=True)

    # Load feature matrix
    rfm_df = pd.read_csv(features_path)
    logger.info(f"Loaded RFM features for {len(rfm_df):,} customers.")

    # Select the scaled features for clustering
    scaled_cols = [
        'Recency_log_scaled', 
        'Frequency_log_scaled', 
        'Monetary_log_scaled', 
        'AvgOrderValue_scaled', 
        'UniqueProducts_scaled', 
        'ReturnRate_scaled',
        'CustomerLifetimeDays_scaled',
        'PurchaseFrequencyMonthly_scaled',
        'AvgQuantityPerOrder_scaled'
    ]
    X = rfm_df[scaled_cols].values

    # ==========================================
    # 1. K-Means Clustering & Evaluation
    # ==========================================
    logger.info("Evaluating K-Means with K in [2, 10]...")
    inertias = []
    sil_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_test = kmeans_test.fit_predict(X)
        inertias.append(kmeans_test.inertia_)
        sil_scores.append(silhouette_score(X, labels_test))
        logger.info(f"K={k} | Inertia: {kmeans_test.inertia_:.2f} | Silhouette Score: {sil_scores[-1]:.4f}")

    # Plot K-Means Elbow & Silhouette
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Elbow Method)', color=color)
    ax1.plot(k_range, inertias, marker='o', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, sil_scores, marker='s', color=color, linewidth=2, linestyle='--', label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('K-Means Clustering Evaluation (Elbow & Silhouette)', fontsize=14, pad=15)
    fig.tight_layout()
    kmeans_eval_plot = os.path.join(plots_dir, 'kmeans_evaluation.png')
    plt.savefig(kmeans_eval_plot, dpi=300)
    plt.close()
    logger.info(f"Saved K-Means evaluation plot to {kmeans_eval_plot}")

    # Train final K-Means with K=4
    logger.info("Fitting final K-Means model with K=4...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    rfm_df['KMeans_Cluster'] = kmeans_labels

    # ==========================================
    # 2. DBSCAN Clustering & EPS Tuning
    # ==========================================
    logger.info("Tuning DBSCAN eps using K-Distance graph (k=5)...")
    # k-distance graph calculation
    k_neigh = 5
    neigh = NearestNeighbors(n_neighbors=k_neigh)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    
    # Sort distances to the 5th nearest neighbor
    sorted_k_distances = np.sort(distances[:, k_neigh - 1])
    
    # Plot K-Distance Graph
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_k_distances, color='purple', linewidth=2)
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k_neigh}-NN Distance')
    plt.title('DBSCAN K-Distance Graph (k=5) for EPS Tuning', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    dbscan_kdist_plot = os.path.join(plots_dir, 'dbscan_k_distance.png')
    plt.savefig(dbscan_kdist_plot, dpi=300)
    plt.close()
    logger.info(f"Saved DBSCAN K-Distance plot to {dbscan_kdist_plot}")

    # Fit DBSCAN with tuned eps (e.g. standard rule is the elbow point, around 1.0 to 1.5 on scaled data)
    # Let's use eps=1.2, min_samples=5 as defaults
    eps_val = 1.2
    min_samples_val = 5
    logger.info(f"Running DBSCAN with eps={eps_val}, min_samples={min_samples_val}...")
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    dbscan_labels = dbscan.fit_predict(X)
    rfm_df['DBSCAN_Cluster'] = dbscan_labels
    
    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_db = list(dbscan_labels).count(-1)
    noise_pct = (n_noise_db / len(X)) * 100
    logger.info(f"DBSCAN Results | Clusters found: {n_clusters_db} | Noise points: {n_noise_db} ({noise_pct:.2f}%)")

    # ==========================================
    # 3. Agglomerative Hierarchical Clustering
    # ==========================================
    logger.info("Generating Dendrogram for Hierarchical Clustering (Ward linkage)...")
    # Generate Linkage Matrix
    Z = linkage(X, method='ward')
    
    # Plot Dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=30,                   # show only the last 30 merged clusters
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True
    )
    plt.axhline(y=45, color='r', linestyle='--', label='Cut Line (K=4)') # illustrative cut threshold
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14)
    plt.xlabel('Cluster Size / Leaf Node')
    plt.ylabel('Distance')
    plt.legend()
    plt.tight_layout()
    dendrogram_plot = os.path.join(plots_dir, 'hierarchical_dendrogram.png')
    plt.savefig(dendrogram_plot, dpi=300)
    plt.close()
    logger.info(f"Saved hierarchical dendrogram to {dendrogram_plot}")

    # Fit Agglomerative Clustering with n_clusters=4
    logger.info("Fitting Agglomerative Hierarchical model with K=4...")
    agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    agg_labels = agg_clustering.fit_predict(X)
    rfm_df['Hierarchical_Cluster'] = agg_labels

    # ==========================================
    # 4. Map K-Means Clusters to Business Labels
    # ==========================================
    logger.info("Profiling K-Means centroids to map business-meaningful labels...")
    
    # Calculate centroids (mean values of raw features for each KMeans cluster)
    centroids = rfm_df.groupby('KMeans_Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    logger.info(f"K-Means Cluster Centroids (Raw):\n{centroids.to_string()}")

    # Determine labels based on centroid values:
    # 1. "Champions": Lowest Recency, highest Frequency and Monetary.
    # 2. "Lost Customers": Highest Recency, lowest Frequency and Monetary.
    # 3. Between the other two:
    #    - The one with higher Recency is "At-Risk" (inactive for a while but has decent frequency/monetary history).
    #    - The one with lower Recency is "Loyal Customers" (frequent, active).
    
    # Let's sort the cluster IDs by Monetary (or Frequency)
    sorted_by_monetary = centroids.sort_values(by='Monetary', ascending=False).index.tolist()
    
    champions_id = sorted_by_monetary[0]
    lost_id = sorted_by_monetary[-1]
    middle_ids = sorted_by_monetary[1:-1]
    
    # For the middle ids, compare Recency (days since last purchase - higher is worse/older)
    if centroids.loc[middle_ids[0], 'Recency'] > centroids.loc[middle_ids[1], 'Recency']:
        at_risk_id = middle_ids[0]
        loyal_id = middle_ids[1]
    else:
        at_risk_id = middle_ids[1]
        loyal_id = middle_ids[0]

    cluster_mapping = {
        champions_id: "Champions",
        loyal_id: "Loyal Customers",
        at_risk_id: "At-Risk",
        lost_id: "Lost Customers"
    }
    
    logger.info(f"Mapped K-Means cluster IDs to labels: {cluster_mapping}")
    
    # Apply mapping
    rfm_df['Segment'] = rfm_df['KMeans_Cluster'].map(cluster_mapping)
    # Also save numeric cluster mapped to standard order (0: Lost, 1: At-Risk, 2: Loyal, 3: Champions)
    segment_orders = {
        "Lost Customers": 0,
        "At-Risk": 1,
        "Loyal Customers": 2,
        "Champions": 3
    }
    rfm_df['Segment_Order'] = rfm_df['Segment'].map(segment_orders)

    # Save final segmented customer data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rfm_df.to_csv(output_path, index=False)
    logger.info(f"Saved final segmented customer data of {len(rfm_df):,} customers to {output_path}")

    # ==========================================
    # 5. Generate 2D PCA Scatter Plot
    # ==========================================
    logger.info("Generating 2D PCA scatter plot of clusters...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    rfm_df['PCA1'] = X_pca[:, 0]
    rfm_df['PCA2'] = X_pca[:, 1]
    
    plt.figure(figsize=(10, 8))
    
    # Palette in standard business styles
    colors = {
        "Champions": "#2ecc71",       # Emerald Green
        "Loyal Customers": "#3498db",  # River Blue
        "At-Risk": "#f1c40f",          # Sun Yellow
        "Lost Customers": "#e74c3c"    # Alizarin Red
    }
    
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Segment', 
        data=rfm_df,
        palette=colors,
        alpha=0.7, 
        edgecolor='k',
        s=40
    )
    
    # Add cluster centroids in PCA space
    pca_centroids = pca.transform(kmeans.cluster_centers_)
    # Map centroids to their segment names to color them correctly
    for idx, (cx, cy) in enumerate(pca_centroids):
        label = cluster_mapping[idx]
        plt.scatter(
            cx, cy, 
            color=colors[label], 
            marker='X', 
            s=250, 
            edgecolor='black', 
            linewidth=2,
            label=f'{label} Centroid' if f'{label} Centroid' not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    plt.title('Customer Segments in 2D PCA Space', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance Explained)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance Explained)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    pca_plot_path = os.path.join(plots_dir, 'customer_segments_pca.png')
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved 2D PCA scatter plot to {pca_plot_path}")
    
    # Log segment distribution
    logger.info(f"Segment Distribution:\n{rfm_df['Segment'].value_counts().to_string()}")

    return rfm_df

if __name__ == "__main__":
    run_segmentation()
