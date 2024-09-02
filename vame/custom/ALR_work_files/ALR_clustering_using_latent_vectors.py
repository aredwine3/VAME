# Clustering using latent vectors

from importlib import reload
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from regex import F

import vame
import vame.custom.ALR_analysis as ana
import vame.custom.ALR_helperFunctions as AlHf
from vame.custom.ALR_latent_vector_cluster_functions import (
    calculate_mean_latent_vector_for_motifs,
    create_tsne_projection,
    create_umap_projection,
    determine_optimal_clusters,
)
from vame.util.auxiliary import read_config

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


#! TRYING WITHOUT SPEED
all_latent_vectors_all_idx, mean_latent_vectors_all_idx = calculate_mean_latent_vector_for_motifs(config, with_speed=False)

vectors = []
labels = []
for idx, latent_vectors in all_latent_vectors_all_idx.items():
	vectors.extend(latent_vectors)
	labels.extend([idx] * len(latent_vectors))
vectors = np.array(vectors)
labels = np.array(labels)


optimal_n_clusters, cluster_labels = determine_optimal_clusters(vectors, clustering_method='kmeans')

#* Labels are the indices of the motifs (ie which motif the latent vector belongs to)
#* Vectors are the mean latent vectors for each motif for each file
#* Cluster labels are the cluster that each vector belongs to


np.shape(all_latent_vectors_all_idx[39]) # (438, 51) # 438 files, 51 dimensions of latent vector (50 + speed)

mean_latent_vectors_all_idx.keys() # 0 to 39

np.shape(labels) # (18142,)

np.shape(vectors) # (18142, 51)
np.shape(cluster_labels) # (18142,)

np.shape(mean_latent_vectors_all_idx[39]) # (51,)

# Get the unique motifs belonging to each cluster
motif_clusters = {cluster: np.where(cluster_labels == cluster)[0] for cluster in np.unique(cluster_labels)}


import umap
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# Visualizing cluster with all_latent_vectors_all_idx, then will try with mean_latent_vectors_all_idx
tsne = TSNE(
    n_components = 3,
    perplexity = 50,
    early_exaggeration=12,
    n_jobs=-1,
)

embedding = tsne.fit_transform(vectors)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    embedding[:, 0], embedding[:, 1], embedding[:, 2], c=cluster_labels, cmap="Spectral", s=5
)

plt.colorbar(
            scatter,
            label="Cluster Index",
            ticks=np.unique(cluster_labels),
            orientation="vertical",
        )

plt.gca().set_aspect("equal", "datalim")
plt.title("t-SNE projection of the Latent Vectors", fontsize=24)

ax.set_title("3D t-SNE projection of the Latent Vectors", fontsize=24)
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_zlabel("t-SNE Component 3")

plt.show()

import matplotlib.pyplot as plt

#### USING UMAP
import umap

# Assuming 'vectors' is your high-dimensional data
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding = reducer.fit_transform(vectors)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    embedding[:, 0], embedding[:, 1], embedding[:, 2], c=cluster_labels, cmap="Spectral", s=5
)

plt.colorbar(scatter, label="Cluster Index", ticks=np.unique(cluster_labels), orientation="vertical")
ax.set_title("3D UMAP projection of the Latent Vectors", fontsize=24)
ax.set_xlabel("UMAP Component 1")
ax.set_ylabel("UMAP Component 2")
ax.set_zlabel("UMAP Component 3")

plt.show()

### Using Hierarchical Clustering
from pathlib import Path

# Using PaCMAP
import pacmap

embedding = pacmap.PaCMAP(n_components=3, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
X_transformed = embedding.fit_transform(vectors, init="pca")

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=cluster_labels, cmap="Spectral", s=5)

ax.set_title("3D PaCMAP projection of the Latent Vectors", fontsize=24)
ax.set_xlabel("PaCMAP Component 1")
ax.set_ylabel("PaCMAP Component 2")
ax.set_zlabel("PaCMAP Component 3")

plt.show()

config_file = Path(config).resolve()
cfg = read_config(config_file)
model_name = cfg['model_name']
n_cluster = cfg['n_cluster']
load_data = cfg['load_data']
hmm_iters = cfg['hmm_iters']
parameterization = cfg['parameterization']

transition_matrices = []
labels = []

files = AlHf.get_files(config)
#labels = get_labels(cfg, files, model_name, n_cluster)

for file in files:
    label = ana.get_label(cfg, file, model_name, n_cluster)
    trans_mat, _ = ana.create_transition_matrix(label, n_cluster)
    labels.append(label)
    transition_matrices.append(trans_mat)

#transition_matrices = compute_transition_matrices(files, labels, n_cluster)

communities_all, motif_to_community_all, clusters_all, clusters_wSpeed_all = ana.community_detection(config,  labels, files, transition_matrices)
aggregated_community_sizes = ana.get_aggregated_community_sizes(files, motif_to_community_all)