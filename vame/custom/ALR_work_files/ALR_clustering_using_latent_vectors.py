# Clustering using latent vectors

from importlib import reload
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

import vame
import vame.custom.ALR_analysis as ana
import vame.custom.ALR_helperFunctions as AlHf
from vame.custom.ALR_latent_vector_cluster_functions import (
    calculate_mean_latent_vector_for_motifs,
    create_tsne_projection,
    create_umap_projection,
)

config = '/work/wachslab/aredwine3/VAME_working/config_sweep_drawn-sweep-88_cpu_hmm-40-650.yaml'

config = 'D:\\Users\\tywin\\VAME\\config.yaml' # * Double backslashes are needed for Windows (\\) to avoid escape characters (The HCC @ UNL uses Linux, but some work is done on a Windows machine)


config = 'D:\\Users\\tywin\\VAME\\config-kmeans-30.yaml'

# Done after manual classification of behaviors
# Add classifications to the master data frame (made during the motif videos creation)

fps = 30

df: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(config, fps=fps, create_new_df=False, df_kind = 'pandas')

df_classifications = pd.read_csv("D:\\Users\\tywin\\VAME\\results\\videos\\hmm-40-650\\ALR_manual_motif_to_behavior_classification_hmm_650.csv")

df = df.merge(df_classifications[['Motif', 'Exclude', 'Moving Quickly', 'Predominant Behavior', 'Secondary Descriptor', 'Category']], 
              left_on='motif', 
              right_on='Motif', 
              how='left')

df.columns = df.columns.str.replace(' ', '_')


all_latent_vectors_all_idx, mean_latent_vectors_all_idx = calculate_mean_latent_vector_for_motifs(config)

print(all_latent_vectors_all_idx.keys())

print(all_latent_vectors_all_idx[1][1])

print(len(all_latent_vectors_all_idx[1]))
print(all_latent_vectors_all_idx[1][1].shape)

create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=False)
create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='agglomerative')
create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='spectral')
create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='gmm')
create_tsne_projection(config, all_latent_vectors_all_idx)

create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='agglomerative')
create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='spectral')
create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='gmm')
create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='kmeans')
create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='dbscan')
create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='hdbscan') 
motif_clusters = create_tsne_projection(config, all_latent_vectors_all_idx, mean_latent_vectors_all_idx, use_cluster_labels=True, use_average_vectors=True, clustering_method='agglomerative')
unique_values = {key: set(value) for key, value in motif_clusters.items()}

mean_latent_vectors_all_idx.keys()
mean_latent_vectors_all_idx.values()
mean_latent_vectors_all_idx[1].shape

# Determine which keys have NaN values
for key, value in mean_latent_vectors_all_idx.items():
    if np.isnan(value).any():
        print(key)

from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

# ! May have to rework how the cluster labels are created
import vame.custom.ALR_analysis as ana
from vame.util.auxiliary import read_config

config_file = Path(config).resolve()
cfg = read_config(config_file)
model_name = cfg['model_name']
n_cluster = cfg['n_cluster']
load_data = cfg['load_data']
hmm_iters = cfg['hmm_iters']
parameterization = cfg['parameterization']

files = AlHf.get_files(config)


mean_transition_matrix = ana.calculate_mean_transition_matrix(cfg, files, model_name, n_cluster)
network = nx.from_numpy_array(mean_transition_matrix)
# Nodes are all_latent_vectors_all_idx.keys()
# Latent vectors (for each motif), are features of each node (columns)
# Edges are the transition probabilities between nodes (ie: Transition Matrix)

# Initialize an empty list to store the feature vectors
features = []

# Loop over the motifs in the dictionary
for motif in sorted(mean_latent_vectors_all_idx.keys()):
    # Get the mean latent vector for this motif
    feature_vector = mean_latent_vectors_all_idx[motif]
    
    # Add the feature vector to the list
    features.append(feature_vector)

# Convert the list of feature vectors to a numpy array
features = np.array(features)

# Convert the feature matrix to a sparse matrix
#features = csr_matrix(features)

# Convert the adjacency matrix to a sparse CSR matri
labels = np.array(list(mean_latent_vectors_all_idx.keys()))
label_mask = np.ones_like(labels)

# Convert the adjacency matrix to a sparse CSR matrix
adj_sparse = csr_matrix(mean_transition_matrix)


# Save the data, indices, and indptr of the sparse CSR matrix in the .npz file
np.savez('C:\\Users\\tywin\\UCODE\\dataset\\redwine_dataset.npz', 
        adj_data=adj_sparse.data, 
        adj_indices=adj_sparse.indices, 
        adj_indptr=adj_sparse.indptr, 
        adj_shape=adj_sparse.shape, 
        attr_matrix=features, 
        labels=labels, 
        label_mask=label_mask)


#! TRYING WITH SPEED
all_latent_vectors_all_idx, mean_latent_vectors_all_idx = calculate_mean_latent_vector_for_motifs(config, with_speed=True)
