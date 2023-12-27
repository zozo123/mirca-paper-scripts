import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import DBSCAN
import numpy as np
import os
import json

# Function to load and process data
def load_and_process_data(dataset):
    pca_file = os.path.join('output', f"{dataset}_clustered_samples_heatmap_pca.csv")
    dca_file = os.path.join('output', f"{dataset}_clustered_samples_heatmap_dca.csv")
    mapping_file = os.path.join('output', f"{dataset}_sample_mapping.json")

    # Load sample name mapping
    with open(mapping_file, 'r') as f:
        sample_mapping = json.load(f)

    # Inverting the mapping for reverse lookup
    inv_sample_mapping = {v: k for k, v in sample_mapping.items()}

    pca_df = pd.read_csv(pca_file, index_col=0)
    dca_df = pd.read_csv(dca_file, index_col=0)

    # Convert integer index back to original sample names
    pca_df.index = [inv_sample_mapping.get(str(idx), idx) for idx in pca_df.index]
    dca_df.index = [inv_sample_mapping.get(str(idx), idx) for idx in dca_df.index]

    # Merge dataframes on 'sample'
    return pd.merge(pca_df, dca_df, left_index=True, right_index=True, suffixes=('_pca', '_dca'))

# Function to perform UMAP embedding and clustering
def umap_and_cluster(data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embedding)
    return embedding, clustering.labels_

# Function to extract detailed identifier from dataset name
def get_detailed_identifier(name):
    parts = name.split('_')
    researcher_year = parts[0]  # First element is the researcher and year
    cancer_type = parts[1]  # Second element is the cancer type
    cell_type = parts[2]  # Third element is the cell type

    # Format cancer type for better readability
    if cancer_type.endswith("Cancer"):
        cancer_type = cancer_type[:-6]  # Remove 'Cancer' from the name

    return f"{researcher_year} {cancer_type} {cell_type}"

# Improve default aesthetics
sns.set(style="whitegrid", palette="viridis", font_scale=1.5)

# Load datasets
datasets = glob.glob(os.path.join('output', "*_clustered_samples_heatmap_pca.csv"))
datasets = [os.path.basename(dataset).split('_clustered_samples_heatmap_pca.csv')[0] for dataset in datasets]

print(datasets)
fig, axs = plt.subplots(len(datasets), 2, figsize=(20, 10 * len(datasets)))

for idx, dataset in enumerate(datasets):
    merged_df = load_and_process_data(dataset)

    # Check for common samples
    if merged_df.empty:
        print(f"No common samples for dataset: {dataset}. Skipping...")
        continue

    pca_data = merged_df.filter(like='_pca').values
    dca_data = merged_df.filter(like='_dca').values

    pca_embedding, pca_clusters = umap_and_cluster(pca_data)
    dca_embedding, dca_clusters = umap_and_cluster(dca_data)

    identifier = get_detailed_identifier(dataset)

    # Visualization
    axs[idx, 0].scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=pca_clusters, cmap='viridis', s=100, edgecolor='k')
    axs[idx, 0].set_title(f"{identifier} PCA Clusters")
    axs[idx, 0].set_xlabel("UMAP 1")
    axs[idx, 0].set_ylabel("UMAP 2")

    axs[idx, 1].scatter(dca_embedding[:, 0], dca_embedding[:, 1], c=dca_clusters, cmap='viridis', s=100, edgecolor='k')
    axs[idx, 1].set_title(f"{identifier} DCA Clusters")
    axs[idx, 1].set_xlabel("UMAP 1")
    axs[idx, 1].set_ylabel("UMAP 2")

# Final adjustments
plt.tight_layout()
plt.savefig("umap_clusters_comparison.png", dpi=100)
plt.show()
