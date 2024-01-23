# Clustering using latent vectors
import vame
import numpy as np
from importlib import reload
import vame.custom.ALR_analysis as ana
from vame.custom.ALR_latent_vector_cluster_functions import calculate_mean_latent_vector_for_motifs

config = '/work/wachslab/aredwine3/VAME_working/config_sweep_drawn-sweep-88_cpu_hmm-40-650.yaml'

all_latent_vectors_all_idx, mean_latent_vectors_all_idx = calculate_mean_latent_vector_for_motifs(config)

print(all_latent_vectors_all_idx.keys())

print(all_latent_vectors_all_idx[1][1])

print(len(all_latent_vectors_all_idx[1]))
print(all_latent_vectors_all_idx[1][1].shape)

ana.create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=False)
ana.create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='agglomerative')
ana.create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='spectral')
ana.create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='gmm')
ana.create_tsne_projection(config, all_latent_vectors_all_idx)

mean_latent_vectors_all_idx.keys()
mean_latent_vectors_all_idx.values()
mean_latent_vectors_all_idx[1].shape

# Determine which keys have NaN values
for key, value in mean_latent_vectors_all_idx.items():
    if np.isnan(value).any():
        print(key)

