import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score
import json
import os

def load_and_process_data(dataset):
    pca_file = os.path.join('output', f"{dataset}_clustered_samples_heatmap_pca.csv")
    dca_file = os.path.join('output', f"{dataset}_clustered_samples_heatmap_dca.csv")
    mapping_file = os.path.join('output', f"{dataset}_sample_mapping.json")

    # Load sample name mapping
    with open(mapping_file, 'r') as f:
        sample_mapping = json.load(f)

    # Inverting the mapping for reverse lookup
    inv_sample_mapping = {v: k for k, v in sample_mapping.items()}

    pca_df = pd.read_csv(pca_file)
    dca_df = pd.read_csv(dca_file)

    # Convert integer sample names back to original sample names
    pca_df['sample'] = pca_df['sample'].map(inv_sample_mapping)
    dca_df['sample'] = dca_df['sample'].map(inv_sample_mapping)

    pca_df['cluster_id_pca'] = pca_df.drop('sample', axis=1).idxmax(axis=1)
    dca_df['cluster_id_dca'] = dca_df.drop('sample', axis=1).idxmax(axis=1)

    return pd.merge(pca_df[['sample', 'cluster_id_pca']], dca_df[['sample', 'cluster_id_dca']], on='sample')

def get_primary_identifier(name):
    parts = name.split('_')
    researcher_year = parts[0]  # First element is the researcher and year
    cancer_type = parts[1]  # Second element is the cancer type
    cell_type = parts[2]  # Third element is the cell type

    # Format cancer type for better readability
    if cancer_type.endswith("Cancer"):
        cancer_type = cancer_type[:-6]  # Remove 'Cancer' from the name

    return f"{researcher_year} {cancer_type} {cell_type}"

def create_heatmap(ax, merged_df, identifier):
    confusion_matrix = pd.crosstab(merged_df['cluster_id_pca'], merged_df['cluster_id_dca'])
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    ax.set_title(get_primary_identifier(identifier), fontsize=16)
    ax.set_ylabel('PCA Clusters')
    ax.set_xlabel('DCA Clusters')
    return ax

datasets = glob.glob(os.path.join('output', "*_clustered_samples_heatmap_pca.csv"))
datasets = [os.path.basename(dataset).split('_clustered_samples_heatmap_pca.csv')[0] for dataset in datasets]

nmi_scores = {}

for dataset in datasets:
    merged_df = load_and_process_data(dataset)
    
    if not merged_df.empty:
        nmi = normalized_mutual_info_score(merged_df['cluster_id_pca'], merged_df['cluster_id_dca'])
        identifier = get_primary_identifier(dataset)
        if identifier in nmi_scores:
            identifier += "_2"
        nmi_scores[identifier] = nmi

with open("nmi_scores.txt", "w") as f:
    for dataset, nmi in nmi_scores.items():
        f.write(f"{dataset}: {nmi}\n")

valid_datasets = [dataset for dataset in datasets if not load_and_process_data(dataset).empty]
fig, axs = plt.subplots(len(valid_datasets), 1, figsize=(10, 10 * len(valid_datasets)))

for idx, dataset in enumerate(valid_datasets):
    merged_df = load_and_process_data(dataset)
    create_heatmap(axs[idx], merged_df, dataset)

fig.tight_layout()
plt.savefig("heatmap_clusters_comparison.png", dpi=100)
plt.show()
