# Data Analysis and Visualization Repository

## Overview

This repository contains scripts for preprocessing data, and preparing visualizations for two specific figures. The scripts are written in Python and utilize libraries such as pandas, matplotlib, seaborn, and scikit-learn.

### Scripts

1. **prep_data.py**
   - Purpose: Data loading and preprocessing.
   - Functions:
     - `load_data(file_path)`: Load data from CSV or TSV files.
     - `remove_all_zero_genes(data)`: Remove genes with all zero values.
     - Additional preprocessing functions.

2. **figure1_prep.py**
   - Purpose: Prepare data and generate visualizations for Figure 1.
   - Features:
     - Data loading and processing.
     - Clustering analysis using DBSCAN.
     - Visualization with UMAP and seaborn.

3. **figure2_prep.py**
   - Purpose: Prepare data and visualizations for Figure 2.
   - Features:
     - Load and process data for PCA and DCA analysis.
     - Generate heatmaps and cluster analysis.
     - Utilizes matplotlib and seaborn for visualization.

## Installation

1. Clone the repository:
   ```
   git clone [repository URL]
   ```

2. Install required packages:
   ```
   pip install pandas matplotlib seaborn scikit-learn umap-learn
   ```
