import csv
import glob
import logging
import os
import pickle
from collections import defaultdict
from importlib import reload
from pathlib import Path
from typing import Dict, List, Tuple

import community as community_louvain
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.signal
import seaborn as sns
import umap
from icecream import ic
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, fcluster, to_tree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    OPTICS,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sympy import SparseMatrix

import vame.custom.ALR_analysis as ana
import vame.custom.ALR_helperFunctions as AlHf
import vame.custom.ALR_plottingFunctions as AlPf
from vame.analysis.pose_segmentation import load_latent_vectors
from vame.custom.ALR_analysis import get_label
from vame.custom.ALR_helperFunctions import get_files
from vame.util.auxiliary import read_config

reload(AlPf)


def calculate_mean_latent_vector_for_motifs(
    config: str,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, np.ndarray]]:
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg["model_name"]
    n_cluster = cfg[
        "n_cluster"
    ]  # the number of motifs (behaviors) identified from the latent vectors
    load_data = cfg["load_data"]  # the dataset used to train the model
    hmm_iters = cfg[
        "hmm_iters"
    ]  # the number of iterations the hidden markov model was trained for
    parameterization = cfg[
        "parameterization"
    ]  # the parameterization used to identify motifs from the latent vectors, hidden markov model or kmeans

    files = AlHf.get_files(config)  # Returns a list of all file names to quantify

    mean_latent_vectors_all_files = {}

    for file in files:
        label = get_label(
            cfg, file, model_name, n_cluster
        )  # Returns a 1D numpy array of the label for each frame (from an open arena video), 'label' refers to the motif that the animal was classified as doing, an integer in the range 0 to n_cluster-1
        resultPath = (
            Path(cfg["project_path"]) / "results" / file / model_name / load_data
        )
        if parameterization == "hmm":
            path = resultPath / f"{parameterization}-{n_cluster}-{hmm_iters}"
        else:
            path = resultPath / f"{parameterization}-{n_cluster}"
        latent_vec = next(path.glob("latent_vector_*.npy"))
        vec = np.load(
            latent_vec
        )  # The latent vector for the given file, a 2D numpy array of shape (n_frames, 50), 50 was the dimensionality of the latent vectors used in the model. n_frames will be identical to the length of "label"
        mean_latent_vectors_file = calculate_mean_vectors(vec, label, n_cluster, path)
        mean_latent_vectors_all_files[file] = mean_latent_vectors_file

    (
        mean_latent_vectors_all_idx,
        all_latent_vectors_all_idx,
    ) = calculate_mean_and_all_latent_vectors_all_motifs(
        mean_latent_vectors_all_files, n_cluster
    )

    return (
        all_latent_vectors_all_idx,
        mean_latent_vectors_all_idx,
    )  # From here I have used the mean_latent_vectors_all_idx to and the clustering functions/methods to visualize the clustering of the latent vectors.


def calculate_mean_vectors(
    vec: np.ndarray, label: np.ndarray, n_cluster: int, path: Path
) -> dict:
    """
    Calculates the mean latent vector for each motif in a given set of latent vectors.

    Args:
        vec (np.ndarray): The set of latent vectors.
        label (np.ndarray): The labels corresponding to each latent vector.
        n_cluster (int): The number of motifs (behaviors) identified from the latent vectors.
        path (Path): The path to save the mean latent vectors.

    Returns:
        mean_latent_vectors_file (dict): A dictionary where the keys are the motif indices and the values are the mean latent vectors for each motif for a single file.
    """
    mean_latent_vectors_file = {}
    for idx in range(n_cluster):
        idx_positions = np.where(label == idx)[0]
        idx_values = vec[idx_positions]
        mean_idx_values = np.mean(idx_values, axis=0)
        mean_latent_vectors_file[idx] = mean_idx_values
        save_mean_vector(path, idx, mean_idx_values)
    return mean_latent_vectors_file


def save_mean_vector(path: str, idx: int, mean_idx_values: np.ndarray) -> None:
    """
    Save the mean latent vector for a given motif to a file.

    Args:
        path (str): The path where the mean latent vector will be saved.
        idx (int): The index of the motif.
        mean_idx_values (np.ndarray): The mean latent vector for the motif.

    Returns:
        None. The function only saves the mean latent vector to a file.
    """
    # Create a folder to save the mean latent vector for this motif in
    mean_vector_path = Path(path) / "mean_latent_vectors_by_motif"
    mean_vector_path.mkdir(exist_ok=True)

    # Save the mean latent vector as a NumPy array
    file_path = mean_vector_path / f"mean_latent_vector_motif_{idx}.npy"
    np.save(file_path, mean_idx_values)


def calculate_mean_and_all_latent_vectors_all_motifs(
    mean_latent_vectors_all_files: Dict[str, Dict[int, List[np.ndarray]]],
    n_cluster: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[np.ndarray]]]:
    """
    Calculates the mean and all latent vectors for each motif (behavior) identified in a set of files.

    Args:
        mean_latent_vectors_all_files: A dictionary containing the mean latent vectors for each file and motif.
            The keys are the file names and the values are dictionaries where the keys are the motif indices
            and the values are lists of numpy arrays representing the mean latent vectors for each motif.
        n_cluster: The number of motifs (behaviors) identified from the latent vectors.

    Returns:
        A tuple containing two dictionaries:
        - mean_latent_vectors_all_idx: A dictionary where the keys are the motif indices and the values are numpy
            arrays representing the mean latent vectors for each motif.
        - all_latent_vectors_all_idx: A dictionary where the keys are the motif indices and the values are lists
            of numpy arrays representing all latent vectors for each motif.
    """
    mean_latent_vectors_all_idx = {idx: [] for idx in range(n_cluster)}
    all_latent_vectors_all_idx = {idx: [] for idx in range(n_cluster)}

    for file_name, mean_vectors_file in mean_latent_vectors_all_files.items():
        for idx, mean_vector in mean_vectors_file.items():
            if np.isnan(mean_vector).any():
                continue
            mean_latent_vectors_all_idx[idx].append(mean_vector)
            all_latent_vectors_all_idx[idx].append(mean_vectors_file[idx])

    for idx, mean_vectors in mean_latent_vectors_all_idx.items():
        mean_latent_vectors_all_idx[idx] = np.mean(mean_vectors, axis=0)

    return mean_latent_vectors_all_idx, all_latent_vectors_all_idx


def cluster_data_with_agglomerative_clustering(n_clusters, vectors, connectivity=True):
    if connectivity:
        from sklearn.neighbors import kneighbors_graph

        connectivity_mat = SparseMatrix()
        connectivity_mat = kneighbors_graph(vectors, n_neighbors=10, include_self=False)

    # Create the agglomerative clustering transformer
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_clusters, connectivity=connectivity_mat
    )

    # Fit the transformer and predict the clusters
    labels = agg_clustering.fit_predict(vectors)

    return labels


def cluster_data_with_spectral_clustering(n_clusters, vectors):
    # Create the spectral clustering transformer
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters, assign_labels="discretize", random_state=0
    )

    # Fit the transformer and predict the clusters
    labels = spectral_clustering.fit_predict(vectors)

    return labels


def cluster_data_with_gmm(n_clusters, vectors):
    # Create the GMM transformer
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)

    # Fit the transformer and predict the clusters
    gmm.fit(vectors)
    labels = gmm.predict(vectors)

    return labels


def cluster_data_with_kmeans(n_clusters, vectors):
    # Create the KMeans transformer
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit the transformer and predict the clusters
    kmeans.fit(vectors)
    labels = kmeans.predict(vectors)

    return labels


def cluster_data_with_dbscan(n_clusters, vectors):
    # Create the DBSCAN transformer
    dbscan = DBSCAN(eps=0.5, min_samples=20)

    # Fit the transformer and predict the clusters
    dbscan.fit(vectors)
    labels = dbscan.labels_

    return labels


def cluster_data_with_optics(n_clusters, vectors):
    # Create the OPTICS transformer
    optics = OPTICS(min_samples=20)

    # Fit the transformer and predict the clusters
    optics.fit(vectors)
    labels = optics.labels_

    return labels


def cluster_data_with_hdbscan(n_clusters, vectors):
    # Create the HDBSCAN transformer
    hdbscan = HDBSCAN(min_cluster_size=15)

    # Fit the transformer and predict the clusters
    hdbscan.fit(vectors)
    labels = hdbscan.labels_

    return labels

def determine_optimal_clusters(vectors, clustering_method="kmeans"):
    silhouette_scores = []  # List to hold silhouette scores for different numbers of clusters
    range_n_clusters = list(range(5, 13))  # The range of cluster numbers to try; adjust as needed

    for n_clusters in range_n_clusters:
        if clustering_method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif clustering_method == "spectral":
            clusterer = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
        elif clustering_method == "gmm":
            clusterer = GaussianMixture(n_components=n_clusters, random_state=0)
        elif clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        elif clustering_method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=20)
        elif clustering_method == "optics":
            clusterer = OPTICS(min_samples=20)
        elif clustering_method == "hdbscan":
            clusterer = HDBSCAN(min_cluster_size=15)
        else:
            raise ValueError("Invalid clustering method")

        cluster_labels = clusterer.fit_predict(vectors)
        silhouette_avg = silhouette_score(vectors, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]

    if clustering_method == "agglomerative":
        labels = cluster_data_with_agglomerative_clustering(optimal_n_clusters, vectors)
    elif clustering_method == "spectral":
        labels = cluster_data_with_spectral_clustering(optimal_n_clusters, vectors)
    elif clustering_method == "gmm":
        labels = cluster_data_with_gmm(optimal_n_clusters, vectors)
    elif clustering_method == "kmeans":
        labels = cluster_data_with_kmeans(optimal_n_clusters, vectors)
    elif clustering_method == "dbscan":
        labels = cluster_data_with_dbscan(optimal_n_clusters, vectors)
    elif clustering_method == "optics":
        labels = cluster_data_with_optics(optimal_n_clusters, vectors)
    elif clustering_method == "hdbscan":
        labels = cluster_data_with_hdbscan(optimal_n_clusters, vectors)

    return optimal_n_clusters, labels

def create_umap_projection(
    config,
    all_latent_vectors_all_idx,
    use_cluster_labels=True,
    clustering_method="agglomerative",
):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg["model_name"]
    n_cluster = cfg["n_cluster"]
    load_data = cfg["load_data"]
    hmm_iters = cfg["hmm_iters"]
    parameterization = cfg["parameterization"]

    if parameterization == "hmm":
        plot_name = f"{parameterization}-{n_cluster}-{hmm_iters}"
    else:
        plot_name = f"{parameterization}-{n_cluster}"

    # Step 2: Prepare the data
    vectors = []
    labels = []
    for idx, latent_vectors in all_latent_vectors_all_idx.items():
        vectors.extend(latent_vectors)
        labels.extend([idx] * len(latent_vectors))
    vectors = np.array(vectors)
    labels = np.array(labels)

    # Remove NaN values
    mask = ~np.isnan(vectors).any(axis=1)
    vectors = vectors[mask]
    labels = labels[mask]

    if use_cluster_labels:
        optimal_n_clusters, labels =  determine_optimal_clusters(vectors, clustering_method)

    # Step 5: Create the UMAP transformer
    reducer = umap.UMAP()

    # Step 6: Fit the transformer and transform the data
    embedding = reducer.fit_transform(vectors)

    # Convert embedding and labels to numpy arrays (if they are not already)
    embedding = np.array(embedding)
    labels = np.array(labels)

    # Step 7: Create the plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=5
    )

    # Step 8: Customize the plot
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection of the Latent Vectors", fontsize=24)

    # Step 9: Add a legend to the right of the plot
    if use_cluster_labels:
        plt.colorbar(
            scatter,
            label="Cluster Index",
            ticks=np.unique(labels),
            orientation="vertical",
        )
        plt.savefig(
            Path(cfg["project_path"])
            / "results"
            / f"umap_projection_{clustering_method}_clusters_{optimal_n_clusters}_{plot_name}.svg",
            dpi=900,
        )

    else:
        plt.colorbar(
            scatter, label="Index", ticks=np.unique(labels), orientation="vertical"
        )
        plt.savefig(
            Path(cfg["project_path"])
            / "results"
            / f"umap_projection_motifs_{plot_name}.svg",
            dpi=900,
        )

    return labels


def create_tsne_projection(
    config,
    all_latent_vectors_all_idx,
    use_cluster_labels=True,
    clustering_method="agglomerative",
):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg["model_name"]
    n_cluster = cfg["n_cluster"]
    load_data = cfg["load_data"]
    hmm_iters = cfg["hmm_iters"]
    parameterization = cfg["parameterization"]

    if parameterization == "hmm":
        plot_name = f"{parameterization}-{n_cluster}-{hmm_iters}"
    else:
        plot_name = f"{parameterization}-{n_cluster}"

    # Step 2: Prepare the data
    vectors = []
    labels = []
    motif_ids = []

    for idx, latent_vectors in all_latent_vectors_all_idx.items():
        vectors.extend(latent_vectors)

        labels.extend([idx] * len(latent_vectors))
        motif_ids.extend([idx] * len(latent_vectors))

    vectors = np.array(vectors)
    labels = np.array(labels)

    # Remove NaN values
    mask = ~np.isnan(vectors).any(axis=1)
    vectors = vectors[mask]
    labels = labels[mask]

    # Step 3: Determine optimal number of clusters
    if use_cluster_labels:
        optimal_n_clusters, labels =  determine_optimal_clusters(vectors, clustering_method)

    # Step 4: Create the t-SNE transformer
    tsne = TSNE(n_components=2, perplexity=40)

    # Step 5: Fit the transformer and transform the data
    embedding = tsne.fit_transform(vectors)

    # Step 6: Create the plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=5
    )

    # Step 7: Customize the plot
    plt.gca().set_aspect("equal", "datalim")
    plt.title("t-SNE projection of the Latent Vectors", fontsize=24)

    # Step 8: Add a legend to the right of the plot
    if use_cluster_labels:
        plt.colorbar(
            scatter,
            label="Cluster Index",
            ticks=np.unique(labels),
            orientation="vertical",
        )
        plt.savefig(
            Path(cfg["project_path"])
            / "results"
            / f"tsne_projection_{clustering_method}_clusters_{optimal_n_clusters}_{plot_name}.svg",
            dpi=900,
        )
    else:
        plt.colorbar(
            scatter, label="Index", ticks=np.unique(labels), orientation="vertical"
        )
        plt.savefig(
            Path(cfg["project_path"])
            / "results"
            / f"tsne_projection_motifs_{plot_name}.svg",
            dpi=900,
        )

    return
