
# ALR_clustering_using_latent_vectors.md


> Note: Done after manual classification of behaviors (done in ALR_making_motif_videos.py)


Import the necessary modules and functions:


```
import vame
import numpy as np
import polars as pl
import pandas as pd
from typing import Union
from importlib import reload
import vame.custom.ALR_analysis as ana
from vame.custom.ALR_latent_vector_cluster_functions import calculate_mean_latent_vector_for_motifs, create_umap_projection, create_tsne_projection
import vame.custom.ALR_helperFunctions as AlHf
```

Define the config file path:
```
config = '/work/wachslab/aredwine3/VAME_working/config_sweep_drawn-sweep-88_cpu_hmm-40-650.yaml'

config = 'D:\\Users\\tywin\\VAME\\config.yaml' # * Double backslashes are needed for Windows (\\) to avoid escape characters (The HCC @ UNL uses Linux, but some work is done on a Windows machine)
```


# 
# Add classifications to the master data frame (made during the motif videos creation)
```
fps = 30

df: Union[pl.DataFrame, pd.DataFrame] = AlHf.create_andOR_get_master_df(config, fps=fps, create_new_df=False, df_kind = 'pandas')

df_classifications = pd.read_csv("D:\\Users\\tywin\\VAME\\results\\videos\\hmm-40-650\\ALR_manual_motif_to_behavior_classification_hmm_650.csv")

df = df.merge(df_classifications[['Motif', 'Moving Quickly', 'Predominant Behavior', 'Category']], 
              left_on='motif', 
              right_on='Motif', 
              how='left')

df.columns = df.columns.str.replace(' ', '_')
```

```
all_latent_vectors_all_idx, mean_latent_vectors_all_idx = calculate_mean_latent_vector_for_motifs(config)

print(all_latent_vectors_all_idx.keys())

print(all_latent_vectors_all_idx[1][1])

print(len(all_latent_vectors_all_idx[1]))
print(all_latent_vectors_all_idx[1][1].shape)

mean_latent_vectors_all_idx.keys()
mean_latent_vectors_all_idx.values()
mean_latent_vectors_all_idx[1].shape


# Determine which keys have NaN values
for key, value in mean_latent_vectors_all_idx.items():
    if np.isnan(value).any():
        print(key)

```

```
create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=False)
create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='agglomerative')
create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='spectral')
create_umap_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='gmm')
create_tsne_projection(config, all_latent_vectors_all_idx)


create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='agglomerative')
create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='spectral')
create_tsne_projection(config, all_latent_vectors_all_idx, use_cluster_labels=True, clustering_method='gmm')
```

{
    0: {0, 3, 6, 7, 8, 9, 10, 14, 19, 21, 24, 26, 27, 29, 30, 32, 34, 35, 37, 38, 39},
    1: {3, 4, 5, 37, 39, 9, 11, 17, 19, 21, 26},
    2: {32, 33, 3, 4, 5, 35, 37, 39, 9, 14, 16, 17, 18, 19, 21, 26, 30, 31},
    3: {2, 38, 39, 9, 19, 22, 23, 24, 27, 29, 31}, 
    4: {3, 4, 5, 35, 7, 37, 9, 39, 14, 16, 17, 18, 19, 21, 26, 28}, 
    5: {32, 33, 3, 35, 36, 6, 38, 39, 9, 10, 14, 16, 19, 26, 30}, 
    6: {32, 33, 34, 3, 35, 5, 6, 7, 38, 10, 13, 14, 15, 16, 18, 21, 26},
    7: {3, 37, 6, 7, 38, 9, 39, 19, 20, 21, 24, 26, 27, 29, 30, 31}, 
    8: {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 24, 25, 26, 28, 32, 34, 35, 37, 38, 39},
    9: {1, 3, 6, 7, 9, 10, 14, 15, 16, 18, 26, 30, 31, 32, 33, 34, 35, 36, 39},
    10: {19, 12, 38},
    11: {33, 3, 35, 5, 37, 39, 9, 19, 26, 27, 29, 30}
}