import glob
import os
import scanpy as sc
import pandas as pd
import numpy as np
import logging
import json

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith(".tsv"):
        data = pd.read_csv(file_path, sep="\t", index_col=0)
    else:
        raise ValueError("Unsupported file format")

    # Normalize sample names to integers
    sample_mapping = {name: idx for idx, name in enumerate(data.columns)}
    data.columns = [sample_mapping[name] for name in data.columns]

    return data, sample_mapping

def remove_all_zero_genes(data):
    cols_to_keep = data.columns[(data.sum(axis=0) > 1e-6)]
    data = data.loc[:, cols_to_keep]

    rows_to_keep = data.index[(data.sum(axis=1) > 1e-6)]
    data = data.loc[rows_to_keep]

    return data

def preprocess_scanpy_data(counts, pre_process=False):
    adata = sc.AnnData(X=counts.T)
    if pre_process:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.log1p(adata)
    return adata

def create_pivot_table(data_df, column_name):
    if "sample" in data_df.index.names:
        data_df = data_df.reset_index()
    pivot_table = data_df.pivot_table(index="sample", columns=column_name, values="cluster", aggfunc="size")
    pivot_table.fillna(0, inplace=True)
    pivot_table = pivot_table.loc[:, pivot_table.sum().sort_values(ascending=False).index]
    return pivot_table

def get_samples_and_clusters(adata, method):
    if method == "pca":
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    elif method == "dca":
        sc.pp.neighbors(adata, use_rep="X_dca")
    sc.tl.leiden(adata)
    clusters = adata.obs["leiden"].astype(str).values
    samples = adata.obs_names
    data_df = pd.DataFrame({"sample": samples, "cluster": clusters})
    return data_df

def remove_all_zero_rows_and_columns(data_counts):
    cols_non_zero = np.any(data_counts > 0, axis=0)
    data_counts = data_counts[:, cols_non_zero]

    rows_non_zero = np.any(data_counts > 0, axis=1)
    data_counts = data_counts[rows_non_zero, :]

    return data_counts

def process_file(file_path):
    name = os.path.basename(file_path).split(".")[0]
    data, sample_mapping = load_data(file_path)  # Updated to receive two values
    # Save sample mapping
    with open(f"{name}_sample_mapping.json", "w") as f:
        json.dump(sample_mapping, f)

    data = remove_all_zero_genes(data)
    num_genes, num_samples = data.shape
    logging.info(f"Processing {name}: Found {num_genes} genes and {num_samples} samples")

    adata = preprocess_scanpy_data(data, pre_process=False)
    adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=None, neginf=None)
    sc.pp.pca(adata, n_comps=min(num_samples, 10))
    adata.obsm["X_pca"] = np.nan_to_num(adata.obsm["X_pca"], nan=0.0)
    logging.info(f"PCA explained variance ratio: {adata.uns['pca']['variance_ratio']}")

    data_df_pca = get_samples_and_clusters(adata, "pca")
    pivot_table_pca = create_pivot_table(data_df_pca, "cluster")
    pivot_table_pca.to_csv(f"./{name}_clustered_samples_heatmap_pca.csv")

    if (adata.X == 0).all(axis=0).any():
        logging.error("All-zero genes detected in adata before DCA. This should not happen after filtering.")
        return

    data_2_exp = np.exp2(data.values.T) - 1
    data_2_exp_clean = np.nan_to_num(data_2_exp, nan=0.0, posinf=1e6, neginf=0)
    data_counts = np.round(data_2_exp_clean).astype(int).astype(float)
    data_counts = np.clip(data_counts, 0, 1e6)

    data_counts = remove_all_zero_rows_and_columns(data_counts)

    adata_raw = sc.AnnData(X=data_counts)
    sc.external.pp.dca(adata_raw, verbose=False, mode="latent", return_info=True, normalize_per_cell=True, early_stop=50)
    adata_raw.obsm["X_dca"] = np.nan_to_num(adata_raw.obsm["X_dca"], nan=0.0)

    data_df_dca = get_samples_and_clusters(adata_raw, "dca")
    pivot_table_dca = create_pivot_table(data_df_dca, "cluster")
    pivot_table_dca.to_csv(f"./{name}_clustered_samples_heatmap_dca.csv")

def main():
    print(os.getcwd())
    files_to_remove = glob.glob("./*heatmap*.csv")

    for file in files_to_remove:
        os.remove(file)

    file_paths_csv = glob.glob("**/*log2*.csv", recursive=True)
    file_paths_tsv = glob.glob("**/*log2*.tsv", recursive=True)
    file_paths = file_paths_csv + file_paths_tsv

    logging.info(f"Found {len(file_paths)} files to process")
    for file_path in file_paths:
        logging.info(f"Processing {file_path}")
        process_file(file_path)
        logging.info(f"Done processing {file_path} and added heatmap files: {file_path}_clustered_samples_heatmap_pca.csv and {file_path}_clustered_samples_heatmap_dca.csv")

        name = os.path.basename(file_path).split(".")[0]
        assert os.path.exists(f"{name}_clustered_samples_heatmap_pca.csv")
        assert os.path.exists(f"{name}_clustered_samples_heatmap_dca.csv")
        assert os.path.exists(f"{name}_sample_mapping.json")

    for file_path in file_paths:
        name = os.path.basename(file_path).split(".")[0]
        if not os.path.exists(f"./{name}_clustered_samples_heatmap_pca.csv"):
            logging.info(f"Missing PCA file for {name}")
        if not os.path.exists(f"./{name}_clustered_samples_heatmap_dca.csv"):
            logging.info(f"Missing DCA file for {name}")

    logging.info("Done")

if __name__ == "__main__":
    main()
