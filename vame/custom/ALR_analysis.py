import csv
import glob
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

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
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, jaccard_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler
from vame.analysis.pose_segmentation import load_latent_vectors
import vame.custom.ALR_helperFunctions as AlHf
import vame.custom.ALR_plottingFunctions as AlPf
from vame.analysis.behavior_structure import get_adjacency_matrix
from vame.analysis.community_analysis import (compute_transition_matrices,
                                              get_labels)
from vame.analysis.tree_hierarchy import (draw_tree, graph_to_tree,
                                          hierarchy_pos, traverse_tree_cutline)
from vame.util.auxiliary import read_config
from importlib import reload
reload(AlPf)

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%I:%M:%S %p'
                    )


def detect_communities_louvain(transition_matrix):
    # Create a graph from the transition matrix
    G = nx.Graph()
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            if transition_matrix[i][j] > 0:
                G.add_edge(i, j, weight=transition_matrix[i][j])

    # Detect communities
    partition = community_louvain.best_partition(G, weight='weight')
    return partition, G


def evaluate_clusters(features, cluster_labels):
    """
    Evaluates the clustering using various internal metrics.

    Parameters:
    features (numpy.ndarray): The array of features that were used for clustering.
    cluster_labels (numpy.ndarray): The array of cluster labels returned by fcluster.

    Returns:
    dict: A dictionary containing the scores from various internal metrics.
    """
    metrics = {}
    
    # Silhouette Score
    if len(set(cluster_labels)) > 1:  # Silhouette score requires at least 2 clusters
        metrics['Silhouette Score'] = silhouette_score(features, cluster_labels)
    
    # Calinski-Harabasz Index
    metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(features, cluster_labels)
    
    # Davies-Bouldin Index
    metrics['Davies-Bouldin Index'] = davies_bouldin_score(features, cluster_labels)
    
    return metrics

def community_detection(config, labels, files, transition_matrices):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    communities_all = []
    motif_to_community_all = [] 
    clusters_all = []
    clusters_wSpeed_all = []
    
    for i, file in enumerate(files):
        partition, G = detect_communities_louvain(transition_matrices[i])
        H = create_hierarchical_graph(partition, transition_matrices[i])

        # Create a hierarchical tree that connects all communities
        hierarchical_tree = create_hierarchical_tree(G, partition)
        
        # Visualization of the network with community coloring
        visualize_communities(G, partition, cfg, file)

        community_trees = create_community_trees(H, partition)
        
        draw_community_tree(config, file, community_trees, method = "community")

        # Draw the hierarchical tree
        draw_community_tree(config, file, hierarchical_tree, method = "hierarchical")

        motif_to_community = {motif: community for motif, community in partition.items()}
        motif_to_community_all.append(motif_to_community)

        communities_all.append(partition)

        label_array = labels[i]

        clusters, clusters_wSpeed = create_clusters(cfg, file, label_array, transition_matrices[i])

        clusters_all.append(clusters)
        clusters_wSpeed_all.append(clusters_wSpeed)
    
    return communities_all, motif_to_community_all, clusters_all, clusters_wSpeed_all


def visualize_communities(G, partition, cfg, file):
    pos = nx.spring_layout(G)  # Position nodes using a layout algorithm
    cmap = plt.get_cmap('viridis', max(partition.values()) + 1)
    hmm_iters = cfg.get('hmm_iters', 0)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    parameterization = cfg['parameterization']

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    for community in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == community]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=100,
                               node_color=[cmap(community)], label=f"Community {community}")
    plt.legend()
    plt.axis('off')
    if parameterization == 'hmm':
        path_to_file = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']) + '-' + str(cfg['hmm_iters']))
    else:
        path_to_file = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']))


    if not os.path.exists(os.path.join(path_to_file, "community_louvain")):
        os.mkdir(os.path.join(path_to_file, "community_louvain"))
    plt.savefig(os.path.join(path_to_file, "community_louvain", f"{file}_communities.png"))
    plt.close('all')


def create_hierarchical_graph(partition, transition_matrix):
    # Create a directed graph
    H = nx.DiGraph()
    
    # Add a super-root node
    super_root = 'Super-Root'
    H.add_node(super_root)
    
    # Find roots for each community and connect them to the super-root
    community_roots = {}
    for node, community in partition.items():
        if community not in community_roots:
            community_roots[community] = node
        else:
            # Example criterion: choose the node with the highest sum of transition probabilities
            if sum(transition_matrix[node]) > sum(transition_matrix[community_roots[community]]):
                community_roots[community] = node
        H.add_edge(super_root, community_roots[community])
    
    # Add edges within each community based on transition probabilities
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            if transition_matrix[i][j] > 0 and partition[i] == partition[j]:
                H.add_edge(community_roots[partition[i]], i)
                H.add_edge(i, j)
    
    return H

def create_hierarchical_tree(G, partition):
    # Create a directed graph for the hierarchical tree
    hierarchical_tree = nx.DiGraph()
    
    # Add a super-root node
    super_root = 'Super-Root'
    hierarchical_tree.add_node(super_root)
    
    # Identify community roots and connect them to the super-root
    community_roots = {}
    for community in set(partition.values()):
        # Find the root node for the current community
        community_nodes = [node for node in partition if partition[node] == community]
        root = max(community_nodes, key=lambda node: G.degree(node))
        community_roots[community] = root
        hierarchical_tree.add_node(root)
        hierarchical_tree.add_edge(super_root, root)
    
    # Connect nodes within each community to their respective roots
    for node, community in partition.items():
        if node != community_roots[community]:  # Avoid connecting the root to itself
            hierarchical_tree.add_node(node)
            hierarchical_tree.add_edge(community_roots[community], node)
    
    return hierarchical_tree

def create_community_trees(H, partition):
    # Create a directed graph for the community trees
    community_trees = nx.DiGraph()
    
    # Identify community roots and construct trees
    for community in set(partition.values()):
        # Find the root node for the current community
        community_nodes = [node for node in partition if partition[node] == community]
        root = max(community_nodes, key=lambda node: H.degree(node))
        
        # Ensure the root is added to community_trees
        community_trees.add_node(root)
        
        # Create a tree for the current community using BFS
        for node in community_nodes:
            # Ensure all nodes are added to community_trees before setting attributes
            community_trees.add_node(node)
            community_trees.nodes[node]['community'] = community
            if node != root:
                # Connect the node to the root of its community
                community_trees.add_edge(root, node)
    
    return community_trees

def draw_community_tree(config, file, community_trees, method = "hierarchical"):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    hmm_iters = cfg.get('hmm_iters', 0)
    load_data = cfg['load_data']
    

    # Check if the graph is empty or has no nodes with in-degree of zero
    if len(community_trees) == 0 or not any(degree == 0 for node, degree in community_trees.in_degree()):
        print(f"No roots found in the graph for file {file}. Cannot draw a hierarchical tree.")
        return  # Exit the function if no roots are found
    
    # Get hierarchical positions
    roots = [node for node, degree in community_trees.in_degree() if degree == 0]

    # If there are multiple roots, you may need to create a super-root or handle them separately
    if len(roots) > 1:
        # Create a super-root and connect it to all roots
        super_root = 'Super-Root'
        community_trees.add_node(super_root)
        for root in roots:
            community_trees.add_edge(super_root, root)
        root_node = super_root
    else:
        root_node = roots[0]

    # Get hierarchical positions
    pos = hierarchy_pos(community_trees, root=root_node)
    
    # Draw the graph
    plt.figure(figsize=(12, 12))
    nx.draw(community_trees, pos, with_labels=True, arrows=False)
    
    if cfg['parameterization'] == 'hmm':
        path_to_file = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']) + '-' + str(cfg['hmm_iters']))
    else:
        path_to_file = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']))
    
    if not os.path.exists(os.path.join(path_to_file, "community_louvain")):
        os.mkdir(os.path.join(path_to_file, "community_louvain"))
    if method == "community":
        plt.savefig(os.path.join(path_to_file, "community_louvain", f"{file}_communities_tree.svg"))
    elif method == "hierarchical":
        plt.savefig(os.path.join(path_to_file, "community_louvain", f"{file}_hierarchical_tree.svg"))

    plt.close()


def get_community_sizes(files, motif_to_community_all):
    for file_idx, motif_to_community in enumerate(motif_to_community_all):
        community_sizes = {}
        for motif, community in motif_to_community.items():
            if community not in community_sizes:
                community_sizes[community] = 0
            community_sizes[community] += 1
        
        # Sort communities by size
        sorted_communities = sorted(community_sizes.items(), key=lambda item: item[1], reverse=True)
        
        print(f"File {file_idx} community sizes:")
        for community, size in sorted_communities:
            print(f"Community {community}: {size} motifs")


def get_aggregated_community_sizes(files, motif_to_community_all):
    aggregated_community_sizes = {}
    
    for motif_to_community in motif_to_community_all:
        for motif, community in motif_to_community.items():
            if community not in aggregated_community_sizes:
                aggregated_community_sizes[community] = 0
            aggregated_community_sizes[community] += 1
    
    return aggregated_community_sizes


def get_motif_duration(label_array):
    """
    Calculate the duration of each motif, which is the count of frames for each motif.
    
    Parameters:
    labels (np.array): An array of motif labels for each frame.
    
    Returns:
    dict: A dictionary with motif identifiers as keys and duration counts as values.
    """
    unique, counts = np.unique(label_array, return_counts=True)
    duration = dict(zip(unique, counts))
    return duration


def get_motif_frequency(label_array):
    """
    Calculate the frequency of each motif, which is the number of times a motif starts.
    
    Parameters:
    labels (np.array): An array of motif labels for each frame.
    
    Returns:
    dict: A dictionary with motif identifiers as keys and frequency counts as values.
    """
    # Find the indices where the motif changes
    change_indices = np.where(np.diff(label_array) != 0)[0] + 1
    # Initialize frequency dictionary
    frequency = {}
    # Add 1 to the count of the first motif since it starts at the first frame
    frequency[label_array[0]] = 1
    # Count the occurrences of each motif at the change points
    for index in change_indices:
        motif = label_array[index]
        if motif in frequency:
            frequency[motif] += 1
        else:
            frequency[motif] = 1
    return frequency


def aggregate_speed_for_motifs(speed_data, label_array):
    avg_speed_per_motif = {}
    
    for motif in np.unique(label_array):
        motif_indices = np.where(label_array == motif)[0]
        avg_speed_per_motif[motif] = np.mean(speed_data[motif_indices])
    
    return avg_speed_per_motif


def calculate_average_speed_for_transitions(n_cluster, label_array, speed_data):
    num_motifs = len(np.unique(label_array))
    if len(np.unique(label_array)) != n_cluster:
        num_motifs = n_cluster
    transition_speed_matrix = np.zeros((num_motifs, num_motifs))
    transition_count_matrix = np.zeros((num_motifs, num_motifs), dtype=int)

    # Example logic for calculating average speed during transitions
    for i in range(1, len(label_array)):
        from_motif = label_array[i - 1]
        to_motif = label_array[i]

        # Only count transitions to a different motif
        if from_motif != to_motif:
            # Assuming speed_data is aligned with label_array
            # and there's a speed value for each frame in label_array
            transition_speed = speed_data[i]

            # Update the transition speed matrix and the count of transitions
            transition_speed_matrix[from_motif, to_motif] += transition_speed
            transition_count_matrix[from_motif, to_motif] += 1

    # Calculate the average by dividing the sum by the count of transitions
    # Use np.divide to handle division by zero, which will result in np.nan for those cases
    average_transition_speed_matrix = np.divide(transition_speed_matrix, transition_count_matrix, out=np.zeros_like(transition_speed_matrix, dtype=float), where=transition_count_matrix!=0)

    return average_transition_speed_matrix


def create_clusters(cfg, file, label_array, transition_matrices):
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']

    if cfg['parameterization'] == 'hmm':
        path = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']) + '-' + str(cfg['hmm_iters']))
    else:
        path = os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], cfg['load_data'], cfg['parameterization'] + '-' + str(cfg['n_cluster']))

    # Initialize avg_speed_per_motif with zeros for all unique motifs
    unique_motifs = np.unique(label_array)
    if len(unique_motifs) != n_cluster:
        unique_motifs = np.arange(n_cluster)
    
    avg_speed_per_motif = {motif: 0 for motif in unique_motifs}

    speed_files = glob.glob(os.path.join(path, "kinematics", f"{file}-*-speed.npy"))
    if speed_files:
        speed_data = np.load(speed_files[0])
        # Update avg_speed_per_motif only for motifs present in speed_data
        for motif in np.unique(label_array):
            indices = np.where(label_array == motif)[0]
            if indices.size > 0:
                avg_speed_per_motif[motif] = np.mean(speed_data[indices])
    
    transition_speed_matrix = calculate_average_speed_for_transitions(n_cluster, label_array, speed_data)

    avg_speed_array = np.array([avg_speed_per_motif[motif] for motif in unique_motifs])

    assert len(avg_speed_per_motif) == len(unique_motifs), "Mismatch in avg_speed_per_motif length"

    matrix_size = transition_matrices.size
    expected_size = n_cluster * (matrix_size // n_cluster)
    
    if matrix_size != expected_size:
        raise ValueError(f"Cannot reshape array of size {matrix_size} into shape ({n_cluster}, -1). Expected size to be a multiple of {n_cluster}.")

    transition_features = transition_matrices.reshape(n_cluster, -1) 
    transition_speeds = transition_speed_matrix.reshape(n_cluster, -1) 
    
    # Create features with transition_speed_matrix
    features_with_speed = np.column_stack([transition_features, transition_speeds, avg_speed_array])
    
    # Create features without transition_speed_matrix
    features_without_speed = np.column_stack([transition_features, avg_speed_array])
    
    # Normalize features with transition_speed_matrix
    scaler_with_speed = StandardScaler()
    normalized_features_with_speed = scaler_with_speed.fit_transform(features_with_speed)
    
    # Normalize features without transition_speed_matrix
    scaler_without_speed = StandardScaler()
    normalized_features_without_speed = scaler_without_speed.fit_transform(features_without_speed)
    
    # Perform hierarchical clustering on features with transition_speed_matrix
    Z_with_speed = sch.linkage(normalized_features_with_speed, method='ward')
    
    # Perform hierarchical clustering on features without transition_speed_matrix
    Z_without_speed = sch.linkage(normalized_features_without_speed, method='ward')

    # Decide on a cutoff threshold and form clusters
    distance_threshold = 10.5  # Adjust based on dendrogram
    clusters = fcluster(Z_without_speed, distance_threshold, criterion='distance')
    clusters_wSpeed = fcluster(Z_with_speed, distance_threshold, criterion='distance')
    
    # Plot the dendrogram for features with transition_speed_matrix
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z_with_speed)
    plt.title('Hierarchical Clustering Dendrogram with Speed')
    plt.xlabel('Motif index')
    plt.ylabel('Distance')
    
    if not os.path.exists(os.path.join(path, "HCD")):
        os.mkdir(os.path.join(path, "HCD"))
    
    plt.savefig(os.path.join(path, "HCD", "HCD_with_speed.svg"))
    plt.close()
    
    # Plot the dendrogram for features without transition_speed_matrix
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z_without_speed)
    plt.title('Hierarchical Clustering Dendrogram without Speed')
    plt.xlabel('Motif index')
    plt.ylabel('Distance')
    
    plt.savefig(os.path.join(path, "HCD", "HCD_without_speed.svg"))
    plt.close()

    if not os.path.exists(os.path.join(path, "kinematics")):
        os.mkdir(os.path.join(path, "kinematics"))
    
    np.save(os.path.join(path, "kinematics", "transition_speed_matrix.npy"), transition_speed_matrix)
    
    # Plot heatmap of transition_speed_matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_speed_matrix, annot=True, annot_kws={"size": 3}, fmt=".2f")
    plt.title('Transition Speed Matrix Heatmap')
    plt.xlabel('Next frame behavior')
    plt.ylabel('Current fram behavior')
    plt.savefig(os.path.join(path, "kinematics", "transition_speed_matrix_heatmap.svg"))
    plt.close()
    
    return clusters, clusters_wSpeed


def calculate_mean_transition_matrix(cfg, file_list, model_name, n_cluster):
    trans_mats = []
    max_motifs = 0

    if not file_list:
        print("No files to process. The file list is empty.")
        return None

    # First pass to determine the maximum number of unique motifs
    for file in file_list:
        label = get_label(cfg, file, model_name, n_cluster)
        if label.size == 0:
            print(f"Warning: No labels found for file {file}. Skipping this file.")
            continue
        max_motifs = max(max_motifs, len(np.unique(label)))
        if max_motifs != n_cluster:
            max_motifs = n_cluster

    if max_motifs == 0:
        print("No motifs found across all files. Cannot create transition matrices.")
        return None

    # Second pass to create transition matrices with uniform size
    for file in file_list:
        label = get_label(cfg, file, model_name, n_cluster)
        if label.size == 0:
            continue  # Skip files with no labels
        trans_mat, motif_to_index = create_transition_matrix(label, max_motifs)
        trans_mats.append(trans_mat)

    if not trans_mats:
        print("No transition matrices were created. The trans_mats list is empty.")
        return None

    # Now we can safely stack the matrices since they all have the same shape
    trans_mat_stack = np.stack(trans_mats, axis=2)
    return np.mean(trans_mat_stack, axis=2)


def get_label(cfg, file, model_name, n_cluster):
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']

    if parameterization == 'hmm':
        path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster)+'-'+str(hmm_iters), "")
    else:
        path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster), "")
    label = np.load(os.path.join(path_to_file,str(n_cluster)+'_km_label_'+file+'.npy'))
    return label


def load_trans_mats(cfg, file_list):
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']

    trans_mats = []
    parameterization = cfg['parameterization']

    for file in file_list:
        if parameterization == 'hmm':
            path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster)+'-'+str(hmm_iters), "")
        else:
            path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster), "")

        trans_mat = np.load(os.path.join(path_to_file, 'behavior_quantification', 'transition_matrix.npy'))
        trans_mats.append(trans_mat)

    return trans_mats


def create_transition_matrix(labels, max_motifs):
    # Find the number of unique motifs and their mapping to zero-based indices
    unique_motifs = np.unique(labels)
    num_motifs = len(unique_motifs)
    motif_to_index = {motif: index for index, motif in enumerate(unique_motifs)}
    
    # Initialize the transition count matrix with zeros
    transition_counts = np.zeros((max_motifs, max_motifs), dtype=int)
    
    # Count transitions from one motif to another
    for i in range(len(labels) - 1):
        current_motif = labels[i]
        next_motif = labels[i + 1]
        if current_motif != next_motif:  # Exclude self-transitions
            # Map motifs to zero-based indices
            current_index = motif_to_index[current_motif]
            next_index = motif_to_index[next_motif]
            transition_counts[current_index, next_index] += 1
    
    # Convert counts to probabilities
    transition_probabilities = np.zeros_like(transition_counts, dtype=float)
    for i in range(num_motifs):
        row_sum = np.sum(transition_counts[i]) - transition_counts[i, i]  # Exclude self-transition count
        if row_sum > 0:
            transition_probabilities[i] = transition_counts[i] / row_sum
            transition_probabilities[i, i] = 0  # Set self-transition probability to zero
    
    return transition_probabilities, motif_to_index


def aggregate_trans_mats(config, imagetype = '.svg'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']
    load_data = cfg['load_data']

    sham_files, injured_files, treated_files, abx_files = AlHf.categorize_fileNames(config)

    file_groups = [sham_files, injured_files, treated_files, abx_files]
    group_names = ["sham", "inj", "treat", "abx"]

    path_to_file = os.path.join(cfg['project_path'], 'results')
    if parameterization == 'hmm':
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters))
    else:
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster))
    os.makedirs(aggregated_analysis_path, exist_ok=True)

    for group_name, file_group in zip(group_names, file_groups):
        transition_speed_matrices = []
        for file in file_group:
            if cfg['parameterization'] == 'hmm':
                path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster) + '-' + str(hmm_iters))
            else:
                path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster))
            
            label_array = get_label(cfg, file, model_name, n_cluster)
            speed_files = glob.glob(os.path.join(path, "kinematics", f"{file}-*-speed.npy"))
            if speed_files:
                speed_data = np.load(speed_files[0])
            else:
                print(f"No speed data file found for {file} in {path}")
                continue
            
            transition_speed_matrix = calculate_average_speed_for_transitions(n_cluster, label_array, speed_data)
            transition_speed_matrices.append(transition_speed_matrix)
        group_trans_speed_mat_stacked = np.stack(transition_speed_matrices, axis=2)
        group_trans_speed_mat_means = np.mean(group_trans_speed_mat_stacked, axis=2)
        np.save(os.path.join(aggregated_analysis_path, f"avg_transition_speed_matrix_{group_name}.npy"), group_trans_speed_mat_means)
        plt.figure(figsize=(12, 12))
        sns.heatmap(group_trans_speed_mat_means, annot=True, annot_kws={"size": 3}, fmt='.2f')
        plt.xlabel("Next frame behavior", fontsize=16)
        plt.ylabel("Current frame behavior", fontsize=16)
        plt.title(f"Averaged Transition Speed matrix of {n_cluster} clusters, Group {group_name}")
        plt.savefig(os.path.join(aggregated_analysis_path, f"avg_transition_speed_matrix_{group_name}{imagetype}"), bbox_inches='tight', transparent=True)
        plt.close()

    trans_mat_means = {
        "sham": calculate_mean_transition_matrix(cfg, sham_files, model_name, n_cluster),
        "inj": calculate_mean_transition_matrix(cfg, injured_files, model_name, n_cluster),
        "treat": calculate_mean_transition_matrix(cfg, treated_files, model_name, n_cluster),
        "abx": calculate_mean_transition_matrix(cfg, abx_files, model_name, n_cluster)
    }

    for group_name, trans_mat_mean in trans_mat_means.items():
        if trans_mat_mean is not None:
            np.save(os.path.join(aggregated_analysis_path, f"avg_transition_matrix_{group_name}.npy"), trans_mat_mean)
            plt.figure(figsize=(12, 12))  # Adjust the size as needed
            sns.heatmap(trans_mat_mean, annot=True, annot_kws={"size": 3}, fmt='.4f')
            plt.xlabel("Next frame behavior", fontsize=16)
            plt.ylabel("Current frame behavior", fontsize=16)
            plt.title(f"Averaged Transition matrix of {n_cluster} clusters, Group {group_name}")
            plt.savefig(os.path.join(aggregated_analysis_path, f"avg_transition_matrix_{group_name}{imagetype}"), bbox_inches='tight', transparent=True)
            plt.close()

            surface_plot = AlPf.plot_3d_transition_heatmap(trans_mat_mean)
            surface_plot.savefig(os.path.join(aggregated_analysis_path, f"avg_transition_matrixSP_{group_name}{imagetype}"), bbox_inches='tight', transparent=True)
            plt.close(surface_plot)

    weeks = set()
    for file_group in file_groups:
        for file in file_group:
            _, study_point, _, _ = AlHf.parse_filename(file)
            weeks.add(study_point)
    
    for week in weeks:
        week_path = os.path.join(aggregated_analysis_path, week)
        os.makedirs(week_path, exist_ok=True)

        for group_name, file_list in zip(group_names, file_groups):
            if week == 'Drug_Trt' and group_name == 'abx':
                continue
            group_week_files = AlHf.get_time_point_columns_for_group(file_list, time_point=week)
            trans_mat_mean = calculate_mean_transition_matrix(cfg, group_week_files, model_name, n_cluster)

            if trans_mat_mean is not None:
                np.save(os.path.join(week_path, f"avg_transition_matrix_{group_name}_{str(week)}.npy"), trans_mat_mean)
                plt.figure(figsize=(12, 12))  # Adjust the size as needed
                sns.heatmap(trans_mat_mean, annot=True, annot_kws={"size": 3}, fmt='.4f')
                plt.xlabel("Next frame behavior", fontsize=16)
                plt.ylabel("Current frame behavior", fontsize=16)
                plt.title(f"Averaged Transition matrix of {n_cluster} clusters, Group {group_name}, {week}")
                plt.savefig(os.path.join(week_path, f"avg_transition_matrix_{group_name}_{week}{imagetype}"), bbox_inches='tight', transparent=True)
                plt.close()

                surface_plot = AlPf.plot_3d_transition_heatmap(trans_mat_mean)
                surface_plot.savefig(os.path.join(week_path, f"avg_transition_matrixSP_{group_name}_{week}{imagetype}"), bbox_inches='tight', transparent=True)
                plt.close(surface_plot)
            
            transition_speed_matrices = []
            
            for file in group_week_files:
                if cfg['parameterization'] == 'hmm':
                    path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster) + '-' + str(hmm_iters))
                else:
                    path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster))
                
                label_array = get_label(cfg, file, model_name, n_cluster)
                speed_files = glob.glob(os.path.join(path, "kinematics", f"{file}-*-speed.npy"))
                if speed_files:
                    speed_data = np.load(speed_files[0])
                else:
                    print(f"No speed data file found for {file} in {path}")
                    continue
                
                transition_speed_matrix = calculate_average_speed_for_transitions(n_cluster, label_array, speed_data)
                transition_speed_matrices.append(transition_speed_matrix)
            if transition_speed_matrices:
                group_trans_speed_mat_stacked = np.stack(transition_speed_matrices, axis=2)
                group_trans_speed_mat_means = np.mean(group_trans_speed_mat_stacked, axis=2)
                np.save(os.path.join(week_path, f"avg_transition_speed_matrix_{group_name}_{str(week)}.npy"), group_trans_speed_mat_means)
                plt.figure(figsize=(12, 12))
                sns.heatmap(group_trans_speed_mat_means, annot=True, annot_kws={"size": 3}, fmt='.2f')
                plt.xlabel("Next frame behavior", fontsize=16)
                plt.ylabel("Current frame behavior", fontsize=16)
                plt.title(f"Averaged Transition Speed matrix of {n_cluster} clusters, Group {group_name}, {week}")
                plt.savefig(os.path.join(week_path, f"avg_transition_speed_matrix_{group_name}_{str(week)}{imagetype}"), bbox_inches='tight', transparent=True)
                plt.close()
                    

def communities(config):
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
        label = get_label(cfg, file, model_name, n_cluster)
        trans_mat, _ = create_transition_matrix(label, n_cluster)
        labels.append(label)
        transition_matrices.append(trans_mat)

    #transition_matrices = compute_transition_matrices(files, labels, n_cluster)
    
    communities_all, motif_to_community_all, clusters_all, clusters_wSpeed_all = community_detection(config,  labels, files, transition_matrices)
    aggregated_community_sizes = get_aggregated_community_sizes(files, motif_to_community_all)
    AlPf.plot_aggregated_community_sizes(config, aggregated_community_sizes)
    
    for idx, file in enumerate(files):
        if parameterization == 'hmm':
            path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster)+'-'+str(hmm_iters))
        else:
            path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster))
        if not os.path.exists(os.path.join(path_to_file,"community")):
            os.mkdir(os.path.join(path_to_file,"community"))
        
        if not os.path.exists(os.path.join(path_to_file,"community_clusters")):
            os.mkdir(os.path.join(path_to_file,"community_clusters"))

        if not os.path.exists(os.path.join(path_to_file, "community_louvain")):
            os.mkdir(os.path.join(path_to_file,"community_louvain"))
        
        np.save(os.path.join(path_to_file,"community_louvain","transition_matrix_"+file+'.npy'),transition_matrices[idx])
        np.save(os.path.join(path_to_file,"community_louvain","motif_to_community_all_"+file+'.npy'), motif_to_community_all[idx])
        np.save(os.path.join(path_to_file,"community_clusters","clusters_all"+file+'.npy'), clusters_all[idx])
        np.save(os.path.join(path_to_file,"community_clusters","clusters_wSpeed_all"+file+'.npy'), clusters_wSpeed_all[idx])
        
        with open(os.path.join(path_to_file,"community","hierarchy"+file+".pkl"), "wb") as fp:   #Pickling
            pickle.dump(communities_all[idx], fp)


def print_properties(obj):
    obj_type = type(obj).__name__
    if obj_type == 'Tensor':
        print("Tensor Properties")
        print(obj)
        print("Shape:", obj.shape)
        print("Size:", obj.size())
        print("Data type:", obj.dtype)
        print("Device:", obj.device)
        print("Requires grad:", obj.requires_grad)
    elif obj_type in ['ndarray', 'memmap']:
        print("Array Properties")
        print(obj)
        print("Shape:", obj.shape)
        print("Size:", obj.size)
        print("Data type:", obj.dtype)
        print("Item size:", obj.itemsize, "bytes")
        print("Nbytes:", obj.nbytes, "bytes")
        print("Number of dimensions:", obj.ndim)
    elif obj_type == 'list':
        print("List Properties")
        print(obj)
        print("Length:", len(obj))
    elif obj_type == 'tuple':
        print("Tuple Properties")
        print(obj)
        print("Length:", len(obj))
    elif obj_type == 'dict':
        print("Dict Properties")
        print(obj)
        print("Keys:", obj.keys())
        print("Values:", obj.values())
        print("Length:", len(obj))
    else:
        print(f"Object of type {obj_type} is not supported.")

def order_dict_keys(input_dict):
    if isinstance(input_dict, dict):
        ordered_dict = dict(sorted(input_dict.items()))
        return ordered_dict
    else:
        raise ValueError("Input must be a dictionary.")


def mean_values_of_dict_list(dict_list):
    # Initialize a dictionary to hold the sum of values for each key
    sum_dict = {}
    
    # Initialize a dictionary to hold the mean values
    mean_dict = {}
    
    # Check if the list is empty
    if not dict_list:
        return mean_dict
    
    # Sum values for each key
    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] = sum_dict.get(key, 0) + value
    
    # Calculate the mean for each key
    num_dicts = len(dict_list)
    for key, sum_value in sum_dict.items():
        mean_dict[key] = sum_value / num_dicts
    
    return mean_dict

def calculate_mean_latent_vector_for_motifs(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']

    files = AlHf.get_files(config)
    
    mean_latent_vectors_all_files = {}
    
    for file in files:
        label = get_label(cfg, file, model_name, n_cluster)
        resultPath = Path(cfg['project_path']) / "results" / file / model_name / load_data
        if parameterization == 'hmm':
            path = resultPath / f"{parameterization}-{n_cluster}-{hmm_iters}"
        else:
            path = resultPath / f"{parameterization}-{n_cluster}"
        latent_vec = next(path.glob('latent_vector_*.npy'))
        vec = np.load(latent_vec)
        mean_latent_vectors_file = calculate_mean_vectors(vec, label, n_cluster, path)
        mean_latent_vectors_all_files[file] = mean_latent_vectors_file
    
    mean_latent_vectors_all_idx, all_latent_vectors_all_idx = calculate_mean_and_all_latent_vectors_all_motifs(mean_latent_vectors_all_files, n_cluster)
    
    return all_latent_vectors_all_idx , mean_latent_vectors_all_idx
    

def calculate_mean_vectors(vec, label, n_cluster, path):
    mean_latent_vectors_file = {}
    for idx in range(n_cluster):
        # Get the positions in label where they are equal to idx
        idx_positions = np.where(label == idx)[0]
        # Get the values of the latent vector at these positions
        idx_values = vec[idx_positions]
        # calculate the mean across the rows (i.e., the first dimension) of idx_values, resulting in a single 50-dimensional vector that is the mean latent vector for the current motif
        mean_idx_values = np.mean(idx_values, axis=0)
        # Add the mean latent vector to the dictionary
        mean_latent_vectors_file[idx] = mean_idx_values
        save_mean_vector(path, idx, mean_idx_values)
    return mean_latent_vectors_file


def save_mean_vector(path, idx, mean_idx_values):
    # Create a folder to save the mean latent vector for this motif in
    mean_vector_path = path / 'mean_latent_vectors_by_motif'
    mean_vector_path.mkdir(exist_ok=True)
    np.save(mean_vector_path / f'mean_latent_vector_motif_{idx}.npy', mean_idx_values)
        
        
def calculate_mean_and_all_latent_vectors_all_motifs(mean_latent_vectors_all_files, n_cluster):
    # Initialize a dictionary to hold the mean latent vectors for each idx
    mean_latent_vectors_all_idx = {idx: [] for idx in range(n_cluster)}
    # Initialize a dictionary to hold all latent vectors for each idx
    all_latent_vectors_all_idx = {idx: [] for idx in range(n_cluster)}
    
    # Iterate over all files
    for _, mean_vectors_file in mean_latent_vectors_all_files.items():
        # For each idx, append the mean latent vector to the list in mean_latent_vectors_all_idx
        for idx, mean_vector in mean_vectors_file.items():
            # Skip this mean_vector if it contains any NaN values
            if np.isnan(mean_vector).any():
                continue
            mean_latent_vectors_all_idx[idx].append(mean_vector)
            all_latent_vectors_all_idx[idx].append(mean_vectors_file[idx])

    # Calculate the mean of the mean latent vectors for each idx
    for idx, mean_vectors in mean_latent_vectors_all_idx.items():
        mean_latent_vectors_all_idx[idx] = np.mean(mean_vectors, axis=0)
        
    return mean_latent_vectors_all_idx, all_latent_vectors_all_idx


def analyze_cluster_data(cfg, avg_motif_duration, avg_motif_frequency, avg_motif_speed, mean_transition_matrix, mean_transition_speed_mat, distance_threshold, find_best_threshold = False):
    if mean_transition_matrix is None:
        raise ValueError("mean_transition_matrix is None, cannot proceed with analysis.")
    n_cluster = cfg['n_cluster']
    num_motifs = len(mean_transition_matrix[0])

    duration_array = np.array([avg_motif_duration.get(i, 0) for i in range(num_motifs)])
    frequency_array = np.array([avg_motif_frequency.get(i, 0) for i in range(num_motifs)])
    speed_array = np.array([avg_motif_speed.get(i, 0) for i in range(num_motifs)])

    assert len(speed_array) == len(duration_array) == len(frequency_array), "Feature lengths do not match"

    transition_features = mean_transition_matrix.reshape(n_cluster, -1)  
    speed_features = mean_transition_speed_mat.reshape(n_cluster, -1) 

    #features = np.column_stack([transition_features, duration_array, frequency_array, speed_array])
    features = np.column_stack([transition_features, speed_array])

    #features_wSpeed = np.column_stack([transition_features, speed_features, duration_array, frequency_array, speed_array])
    features_wSpeed = np.column_stack([transition_features, speed_features, speed_array])

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    normalized_features_wSpeed = scaler.fit_transform(features_wSpeed)

    Z = sch.linkage(normalized_features, method='ward', optimal_ordering=True)
    ZwSpeed = sch.linkage(normalized_features_wSpeed, method='ward', optimal_ordering=True)
    
    if find_best_threshold:
        best_threshold = find_best_distance_threshold(Z, normalized_features, 0.5, 15, 30)
        print(f"Best distance threshold: {best_threshold}")
        distance_threshold = best_threshold
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))

    sch.dendrogram(Z, color_threshold=distance_threshold, ax=axs[0])
    axs[0].axhline(y=distance_threshold, color='r', linestyle='--')
    axs[0].set_title('Dendrogram without Speed Features')
    axs[0].set_xlabel('Motif Index')
    axs[0].set_ylabel('Distance')

    sch.dendrogram(ZwSpeed, color_threshold=distance_threshold, ax=axs[1])
    axs[1].axhline(y=distance_threshold, color='r', linestyle='--')
    axs[1].set_title('Dendrogram with Speed Features')
    axs[1].set_xlabel('Motif Index')
    axs[1].set_ylabel('Distance')

    clusters = fcluster(Z, distance_threshold, criterion='distance')
    clusters_wSpeed = fcluster(ZwSpeed, distance_threshold, criterion='distance')

    return clusters, clusters_wSpeed, normalized_features, normalized_features_wSpeed, fig



def find_best_distance_threshold(Z, features, min_threshold, max_threshold, num_thresholds):
    """
    Finds the best distance threshold for hierarchical clustering based on internal metrics.

    Parameters:
    Z (numpy.ndarray): The linkage matrix from hierarchical clustering.
    features (numpy.ndarray): The array of features that were used for clustering.
    min_threshold (float): The minimum distance threshold to try.
    max_threshold (float): The maximum distance threshold to try.
    num_thresholds (int): The number of thresholds to try between the minimum and maximum.

    Returns:
    float: The best distance threshold based on the average of internal metrics.
    """
    best_threshold = min_threshold
    best_score = -float('inf')
    
    # Generate a range of distance thresholds to try
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    
    for threshold in thresholds:
        # Form clusters using the current threshold
        cluster_labels = fcluster(Z, threshold, criterion='distance')
        
        # Skip the evaluation if there's only one cluster or each sample is its own cluster
        num_unique_clusters = len(set(cluster_labels))
        if num_unique_clusters <= 1 or num_unique_clusters == len(cluster_labels):
            continue
        
        # Evaluate clusters using internal metrics
        try:
            silhouette_avg = silhouette_score(features, cluster_labels)
        except ValueError:
            silhouette_avg = -1  # Silhouette score is not valid for one cluster or one cluster per sample
        
        calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
        davies_bouldin = davies_bouldin_score(features, cluster_labels)
        
        # Calculate the average score across metrics
        # Note: Davies-Bouldin index should be minimized, so it's subtracted
        average_score = (silhouette_avg + calinski_harabasz - davies_bouldin) / 3
        
        # Update the best threshold if the current one is better
        if average_score > best_score:
            best_score = average_score
            best_threshold = threshold
    
    return best_threshold


def compare_weekly_clusters_OG(weekly_clusters):
    """
    Compares weekly clusters and calculates the Adjusted Rand Index (ARI), Jaccard Index,
    and cluster transition counts between each pair of weeks.

    Parameters:
    weekly_clusters (dict): A dictionary where keys are weeks and values are cluster assignments.

    Returns:
    dict: A dictionary of comparison metrics for each pair of weeks.
    """
    weeks = ["Baseline_1", "Baseline_2", "Week_02", "Week_04", "Week_06", "Week_08", "Week_11", "Week_13", "Week_15", "Drug_Trt"]
    #weeks = sorted(weekly_clusters.keys())
    comparison_metrics = {}
    for i in range(len(weeks) - 1):
        for j in range(i + 1, len(weeks)):
            week_i = weeks[i]
            week_j = weeks[j]
            clusters_i = weekly_clusters[week_i]
            clusters_j = weekly_clusters[week_j]

            # Map cluster IDs to zero-based indices for both weeks
            unique_clusters_i = np.unique(clusters_i)
            unique_clusters_j = np.unique(clusters_j)
            cluster_to_index_i = {cluster_id: index for index, cluster_id in enumerate(unique_clusters_i)}
            cluster_to_index_j = {cluster_id: index for index, cluster_id in enumerate(unique_clusters_j)}

            # Calculate ARI
            ari = adjusted_rand_score(clusters_i, clusters_j)

            # Calculate Jaccard Indices
            jaccard_indices = []
            for cluster_id in unique_clusters_i:
                members_i = set(np.where(clusters_i == cluster_id)[0])
                members_j = set(np.where(clusters_j == cluster_id)[0])
                intersection = len(members_i.intersection(members_j))
                union = len(members_i.union(members_j))
                jaccard_indices.append(intersection / union if union > 0 else 0)

            # Initialize transition counts matrix
            transition_counts = np.zeros((len(unique_clusters_i), len(unique_clusters_j)))

            # Calculate cluster transition counts
            for idx, (ci, cj) in enumerate(zip(clusters_i, clusters_j)):
                index_i = cluster_to_index_i[ci]
                index_j = cluster_to_index_j[cj]
                transition_counts[index_i, index_j] += 1

            # Store the metrics
            comparison_metrics[(week_i, week_j)] = {
                'ARI': ari,
                'Jaccard Indices': jaccard_indices,
                'Transition Counts': transition_counts
            }
    return comparison_metrics


def compare_weekly_clusters(weekly_clusters):
    weeks = ["Baseline_1", "Baseline_2", "Week_02", "Week_04", "Week_06", "Week_08", "Week_11", "Week_13", "Week_15", "Drug_Trt"]
    comparison_metrics = {}
    ari_scores = []
    jaccard_indices = []
    transition_matrices = []

     # Initialize lists to store the dimensions of each transition matrix
    num_rows_list = []
    num_cols_list = []

        # Find the maximum number of clusters for any week
    max_rows = max_cols = 0
    for week in weeks:
        unique_clusters = np.unique(weekly_clusters[week])
        max_rows = max(max_rows, len(unique_clusters))
        max_cols = max(max_cols, len(unique_clusters))

    for i in range(len(weeks) - 1):
        for j in range(i + 1, len(weeks)):
            week_i = weeks[i]
            week_j = weeks[j]
            clusters_i = weekly_clusters[week_i]
            clusters_j = weekly_clusters[week_j]

            # Calculate ARI
            ari = adjusted_rand_score(clusters_i, clusters_j)
            ari_scores.append(ari)

            # Calculate Jaccard Indices
            jaccard_indices_week = []
            for cluster_id in np.unique(clusters_i):
                members_i = set(np.where(clusters_i == cluster_id)[0])
                members_j = set(np.where(clusters_j == cluster_id)[0])
                intersection = len(members_i.intersection(members_j))
                union = len(members_i.union(members_j))
                jaccard_index = intersection / union if union > 0 else 0
                jaccard_indices_week.append(jaccard_index)
            jaccard_indices.extend(jaccard_indices_week)


            # Calculate transition counts
            transition_counts_matrix = np.zeros((max_rows, max_cols))
            for idx, (ci, cj) in enumerate(zip(clusters_i, clusters_j)):
                index_i = np.where(np.unique(clusters_i) == ci)[0][0]
                index_j = np.where(np.unique(clusters_j) == cj)[0][0]
                transition_counts_matrix[index_i, index_j] += 1

            # Store the dimensions of the transition matrix
            num_rows_list.append(transition_counts_matrix.shape[0])
            num_cols_list.append(transition_counts_matrix.shape[1])
            
            transition_matrices.append(transition_counts_matrix)

            # Store the metrics
            comparison_metrics[(week_i, week_j)] = {
                'ARI': ari,
                'Jaccard Indices': jaccard_indices_week,
                'Transition Counts': transition_counts_matrix
            }

    # Calculate the variance in the number of rows and columns
    rows_variance = np.var(num_rows_list)
    cols_variance = np.var(num_cols_list)

    # Calculate mean and standard deviation for ARI scores
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)

    # Calculate mean and standard deviation for Jaccard indices
    jaccard_mean = np.mean(jaccard_indices)
    jaccard_std = np.std(jaccard_indices)

    # Calculate mean and standard deviation for transition counts
    transition_counts_mean = np.mean(transition_matrices, axis=0)
    transition_counts_std = np.std(transition_matrices, axis=0)

    # Add the statistics to the start of the comparison metrics dictionary
    comparison_metrics['Clustering Statistics'] = {
        'ARI Mean': ari_mean,
        'ARI Std': ari_std,
        'Jaccard Mean': jaccard_mean,
        'Jaccard Std': jaccard_std,
        'Transition Mean': transition_counts_mean,
        'Transition Std': transition_counts_std
    }
        # Add the variance in dimensions to the clustering statistics
    comparison_metrics['Clustering Statistics']['Rows Variance'] = rows_variance
    comparison_metrics['Clustering Statistics']['Cols Variance'] = cols_variance

    return comparison_metrics


def save_cluster_comparison_metrics_csv(comparison_metrics, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Week Pair', 'ARI', 'Jaccard Indices', 'Transition Counts', 'ARI Mean', 'ARI Std', 'Jaccard Mean', 'Jaccard Std', 'Transition Mean', 'Transition Std', 'Rows Variance', 'Cols Variance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for week_pair, metrics in comparison_metrics.items():
            if week_pair == 'Clustering Statistics':
                continue  # Skip the overall statistics for individual week pair entries
            row = {
                'Week Pair': f"{week_pair[0]} vs {week_pair[1]}",
                'ARI': metrics['ARI'],
                'Jaccard Indices': metrics['Jaccard Indices'],
                'Transition Counts': metrics['Transition Counts'].tolist()  # Convert numpy array to list
            }
            writer.writerow(row)
        
        # Write the overall clustering statistics at the end of the CSV
        clustering_stats = comparison_metrics.get('Clustering Statistics', {})
        if clustering_stats:
            row = {
                'Week Pair': 'Overall Statistics',
                'ARI Mean': clustering_stats.get('ARI Mean'),
                'ARI Std': clustering_stats.get('ARI Std'),
                'Jaccard Mean': clustering_stats.get('Jaccard Mean'),
                'Jaccard Std': clustering_stats.get('Jaccard Std'),
                'Transition Mean': clustering_stats.get('Transition Mean', np.array([])).tolist(),
                'Transition Std': clustering_stats.get('Transition Std', np.array([])).tolist(),
                'Rows Variance': clustering_stats.get('Rows Variance'),
                'Cols Variance': clustering_stats.get('Cols Variance')
            }
            writer.writerow(row)


def save_cluster_comparison_metrics_txt(comparison_metrics, filename):
    with open(filename, 'w') as txtfile:
        for week_pair, metrics in comparison_metrics.items():
            if week_pair == 'Clustering Statistics':
                continue  # Skip the overall statistics for individual week pair entries
            txtfile.write(f"Metrics between {week_pair[0]} and {week_pair[1]}:\n")
            txtfile.write(f"ARI: {metrics['ARI']}\n")
            txtfile.write(f"Jaccard Indices: {metrics['Jaccard Indices']}\n")
            txtfile.write(f"Transition Counts:\n{metrics['Transition Counts']}\n\n")
        
        # Write the overall clustering statistics at the end of the TXT file
        clustering_stats = comparison_metrics.get('Clustering Statistics', {})
        if clustering_stats:
            txtfile.write("Overall Clustering Statistics:\n")
            txtfile.write(f"ARI Mean: {clustering_stats.get('ARI Mean')}\n")
            txtfile.write(f"ARI Std: {clustering_stats.get('ARI Std')}\n")
            txtfile.write(f"Jaccard Mean: {clustering_stats.get('Jaccard Mean')}\n")
            txtfile.write(f"Jaccard Std: {clustering_stats.get('Jaccard Std')}\n")
            txtfile.write(f"Transition Mean: {np.array2string(clustering_stats.get('Transition Mean', np.array([])))}\n")
            txtfile.write(f"Transition Std: {np.array2string(clustering_stats.get('Transition Std', np.array([])))}\n")
            txtfile.write(f"Rows Variance: {clustering_stats.get('Rows Variance')}\n")
            txtfile.write(f"Cols Variance: {clustering_stats.get('Cols Variance')}\n")


def calculate_clustering_statistics_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    ari_scores = []
    jaccard_indices = []
    transition_counts = []

    for line in lines:
        if line.startswith('ARI:'):
            ari_scores.append(float(line.split(':')[1].strip()))
        elif line.startswith('Jaccard Indices:'):
            indices = eval(line.split(':')[1].strip())
            jaccard_indices.extend(indices)
        elif line.startswith('Transition Counts:'):
            # The next few lines contain the transition counts matrix
            matrix_start_index = lines.index(line) + 1
            matrix_end_index = matrix_start_index
            while lines[matrix_end_index].strip():
                matrix_end_index += 1
            # Extract the matrix and flatten it
            transition_matrix = np.array([
                np.fromstring(row.strip("[]\n"), sep=' ') for row in lines[matrix_start_index:matrix_end_index]
            ])
            transition_counts.extend(transition_matrix.flatten())

    # Calculate mean and standard deviation for ARI scores
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)

    # Calculate mean and standard deviation for Jaccard indices
    jaccard_mean = np.mean(jaccard_indices)
    jaccard_std = np.std(jaccard_indices)

    # Calculate mean and standard deviation for transition counts
    transition_mean = np.mean(transition_counts)
    transition_std = np.std(transition_counts)

    return {
        'ARI Mean': ari_mean,
        'ARI Std': ari_std,
        'Jaccard Mean': jaccard_mean,
        'Jaccard Std': jaccard_std,
        'Transition Mean': transition_mean,
        'Transition Std': transition_std
    }


def clustering_differences_across_time(config, distance_threshold = 2, imagetype = '.svg'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']

    files = AlHf.get_files(config)    

    weeks = set()

    for file in files:
        _, study_point, _, _ = AlHf.parse_filename(file)
        weeks.add(study_point)

    weekly_motif_durations = {}
    weekly_motif_frequencies = {}
    weekly_avg_speed_per_motif_list = {}
    weekly_mean_trans_mats = {}
    weekly_mean_transition_speed_mats = {}
    weekly_clusters = {}
    weekly_features = {}
    weekly_clusters_wSpeed = {}
    weekly_features_wSpeed = {}

    path_to_file = os.path.join(cfg['project_path'], 'results')
    if parameterization == 'hmm':
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters))
    else:
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster))

    for week in weeks:
        week_path = os.path.join(aggregated_analysis_path, week)
        os.makedirs(week_path, exist_ok=True)
        
        week_files = AlHf.get_time_point_columns_for_group(files, time_point=week)

        labels = []
        motif_durations = []
        motif_frequencies = []
        avg_speed_per_motif_list = []
        transition_speed_mats = []

        for file in week_files:
            if parameterization == 'hmm':
                path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster) + '-' + str(hmm_iters))
            else:
                path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster))

            label_array = get_label(cfg, file, model_name, n_cluster)
            motif_duration = get_motif_duration(label_array)
            motif_frequency = get_motif_frequency(label_array)

            speed_files = glob.glob(os.path.join(path, "kinematics", f"{file}-*-speed.npy"))
            if speed_files:
                speed_data = np.load(speed_files[0])
            else:
                print(f"No speed data file found for {file} in {path}")
                continue
            
            transition_speed_matrix = calculate_average_speed_for_transitions(n_cluster, label_array, speed_data)
            avg_speed_per_motif = aggregate_speed_for_motifs(speed_data, label_array)
            motif_duration_ordered = order_dict_keys(motif_duration)
            motif_frequency_ordered = order_dict_keys(motif_frequency)
            avg_speed_per_motif_ordered = order_dict_keys(avg_speed_per_motif)
            
            labels.append(label_array)
            motif_durations.append(motif_duration_ordered)
            motif_frequencies.append(motif_frequency_ordered)
            avg_speed_per_motif_list.append(avg_speed_per_motif_ordered)
            transition_speed_mats.append(transition_speed_matrix)

        avg_motif_duration = mean_values_of_dict_list(motif_durations)
        avg_motif_frequency = mean_values_of_dict_list(motif_frequencies)
        avg_motif_speed = mean_values_of_dict_list(avg_speed_per_motif_list)
        mean_transition_matrix = calculate_mean_transition_matrix(cfg, week_files, model_name, n_cluster)
        stacked_speed_mats = np.stack(transition_speed_mats, axis=2)
        mean_transition_speed_matrix = np.mean(stacked_speed_mats, axis=2)


        plt.figure(figsize=(12, 12))  # Adjust the size as needed
        sns.heatmap(mean_transition_matrix, annot=True, annot_kws={"size": 3}, fmt='.4f')
        plt.xlabel("Next frame behavior", fontsize=16)
        plt.ylabel("Current frame behavior", fontsize=16)
        plt.title(f"Averaged Transition matrix of {n_cluster} clusters, All Data")
        plt.savefig(os.path.join(week_path, f"avg_transition_matrix_{week}_{distance_threshold}{imagetype}"), bbox_inches='tight', transparent=True)
        plt.close()

        plt.figure(figsize=(12, 12))  # Adjust the size as needed
        sns.heatmap(mean_transition_speed_matrix, annot=True, annot_kws={"size": 3}, fmt='.2f')
        plt.xlabel("Next frame behavior", fontsize=16)
        plt.ylabel("Current frame behavior", fontsize=16)
        plt.title(f"Averaged Transition Speed matrix of {n_cluster} clusters, All Data")
        plt.savefig(os.path.join(week_path, f"avg_transition_speed_matrix_{week}_{distance_threshold}{imagetype}"), bbox_inches='tight', transparent=True)
        plt.close()

        weekly_motif_durations[week] = avg_motif_duration
        weekly_motif_frequencies[week] = avg_motif_frequency
        weekly_avg_speed_per_motif_list[week] = avg_motif_speed
        weekly_mean_trans_mats[week] = mean_transition_matrix
        weekly_mean_transition_speed_mats[week] = mean_transition_speed_matrix

        clusters, clusters_wSpeed, normalized_features, normalized_features_wSpeed, _ =  analyze_cluster_data(cfg, avg_motif_duration, avg_motif_frequency, avg_motif_speed, mean_transition_matrix, mean_transition_speed_matrix, distance_threshold, find_best_threshold = False)
        
        weekly_clusters[week] = clusters
        weekly_features[week] = normalized_features

        weekly_clusters_wSpeed[week] = clusters_wSpeed
        weekly_features_wSpeed[week] = normalized_features_wSpeed


        np.save(os.path.join(week_path, f"avg_motif_duration-{week}.npy"), list(avg_motif_duration.values()))
        np.save(os.path.join(week_path, f"avg_motif_frequency-{week}.npy"), list(avg_motif_frequency.values()))
        np.save(os.path.join(week_path, f"avg_motif_speed-{week}.npy"), list(avg_motif_speed.values()))
        
        if mean_transition_matrix is not None:
            np.save(os.path.join(week_path, f"mean_transition_matrix-{week}.npy"), mean_transition_matrix)
        else:
            print(f"Warning: mean_transition_matrix is None for {week}, not saving this file.")
            
        np.save(os.path.join(week_path, f"mean_transition_speed_matrix-{week}.npy"), mean_transition_speed_matrix)


    matched_clusters = match_and_relabel_clusters(weekly_clusters)
    matched_clusters_wSpeed = match_and_relabel_clusters(weekly_clusters_wSpeed)

    # Save weekly clusters to a file
    np.save(os.path.join(aggregated_analysis_path, f"weekly_clusters.npy"), weekly_clusters)
    np.save(os.path.join(aggregated_analysis_path, f"weekly_clusters_wSpeed.npy"), weekly_clusters_wSpeed)
    
    for week in weeks:
        week_path = os.path.join(aggregated_analysis_path, week)
        os.makedirs(week_path, exist_ok=True)
        
        HCD_path = os.path.join(week_path, "HCD")
        os.makedirs(HCD_path, exist_ok=True)
        
        _, _, _, _, fig = analyze_cluster_data(cfg, weekly_motif_durations[week], weekly_motif_frequencies[week], weekly_avg_speed_per_motif_list[week], weekly_mean_trans_mats[week], weekly_mean_transition_speed_mats[week], distance_threshold)
        
        fig.savefig(os.path.join(week_path, "HCD", f"HCD-{week}-{distance_threshold}{imagetype}"))
        plt.close(fig)        

        if week in matched_clusters:
            clusters = matched_clusters[week]
            ic(clusters)
            np.save(os.path.join(week_path, f"clusters-{week}-{distance_threshold}.npy"), clusters)
            fig = AlPf.plot_motif_statistics(weekly_motif_durations[week], weekly_motif_frequencies[week], weekly_avg_speed_per_motif_list[week], clusters)
            fig.savefig(os.path.join(week_path, f"motif_statistics_by_cluster-{week}-{distance_threshold}{imagetype}"), bbox_inches='tight')
            plt.close(fig)

        else:
            print(f"Warning: No matching clusters found for {week}.")

        if week in matched_clusters_wSpeed:
            clusters_wSpeed = matched_clusters_wSpeed[week]
            ic(clusters_wSpeed)
            np.save(os.path.join(week_path, f"clusters_wSpeed-{week}-{distance_threshold}.npy"), clusters_wSpeed)
            fig = AlPf.plot_motif_statistics(weekly_motif_durations[week], weekly_motif_frequencies[week], weekly_avg_speed_per_motif_list[week], clusters_wSpeed)
            fig.savefig(os.path.join(week_path, f"motif_statistics_by_cluster_wSpeed-{week}-{distance_threshold}{imagetype}"), bbox_inches='tight')
            plt.close(fig)

        else:
            print(f"Warning: No matching clusters with speed found for {week}.")
    
    # Function to save plots and close figures
    def save_and_close_plot(plot, filename):
        plot.savefig(filename, bbox_inches='tight')
        plt.close(plot)

    comparison_metrics = compare_weekly_clusters(matched_clusters)
    comparison_metrics_wSpeed = compare_weekly_clusters(matched_clusters_wSpeed)

    comparison_metrics_plot = compare_weekly_clusters_OG(matched_clusters)
    comparison_metrics_wSpeed_plot = compare_weekly_clusters_OG(matched_clusters_wSpeed)
    
    save_cluster_comparison_metrics_csv(comparison_metrics, os.path.join(aggregated_analysis_path, f"weekly_cluster_comparison_metrics-{distance_threshold}.csv"))
    save_cluster_comparison_metrics_txt(comparison_metrics, os.path.join(aggregated_analysis_path, f"weekly_cluster_comparison_metrics-{distance_threshold}.txt"))

    save_cluster_comparison_metrics_csv(comparison_metrics_wSpeed, os.path.join(aggregated_analysis_path, f"weekly_cluster_wSpeed_comparison_metrics-{distance_threshold}.csv"))
    save_cluster_comparison_metrics_txt(comparison_metrics_wSpeed, os.path.join(aggregated_analysis_path, f"weekly_cluster_wSpeed_comparison_metrics-{distance_threshold}.txt"))

    # Generate and save plots for cluster comparisons
    weeks = sorted(set(week for week_pair in comparison_metrics_plot for week in week_pair))
    save_and_close_plot(AlPf.create_cluster_metric_heatmap(comparison_metrics_plot, weeks, 'ARI'), os.path.join(aggregated_analysis_path, f"ari_plot-cluster_comparisons-{distance_threshold}{imagetype}"))
    save_and_close_plot(AlPf.plot_3d_jaccard_indices(comparison_metrics_plot, weeks), os.path.join(aggregated_analysis_path, f"jaccard_plot-cluster_comparisons-{distance_threshold}{imagetype}"))

    weeks = sorted(set(week for week_pair in comparison_metrics_wSpeed_plot for week in week_pair))
    save_and_close_plot(AlPf.create_cluster_metric_heatmap(comparison_metrics_wSpeed_plot, weeks, 'ARI'), os.path.join(aggregated_analysis_path, f"ari_plot-cluster_wSpeed_comparisons-{distance_threshold}{imagetype}"))
    save_and_close_plot(AlPf.plot_3d_jaccard_indices(comparison_metrics_wSpeed_plot, weeks), os.path.join(aggregated_analysis_path, f"jaccard_plot-cluster_wSpeed_comparisons-{distance_threshold}{imagetype}"))


def match_and_relabel_clusters(weekly_clusters):
    weeks = sorted(weekly_clusters.keys())
    # Step 1: Determine motifs in clusters for each time point
    motif_cluster_mapping = {week: {cluster: np.where(clusters == cluster)[0].tolist()
                                    for cluster in np.unique(clusters)}
                             for week, clusters in weekly_clusters.items()}

    # Initialize the new weekly clusters with the first week's clusters
    new_weekly_clusters = {weeks[0]: weekly_clusters[weeks[0]]}

    # Step 2: Determine the most similar clusters between consecutive weeks
    weeks = sorted(weekly_clusters.keys())
    for i in range(len(weeks) - 1):
        week1, week2 = weeks[i], weeks[i + 1]
        clusters1 = motif_cluster_mapping[week1]
        clusters2 = motif_cluster_mapping[week2]

        # Find the most similar clusters between the two weeks
        similar_clusters = find_most_similar_clusters(clusters1, clusters2)

        # Update the cluster labels for week2 to match the most similar clusters from week1
        updated_clusters_week2 = update_cluster_labels(clusters1, clusters2, similar_clusters)

        # Store the updated labels in the new weekly clusters
        new_weekly_clusters[week2] = updated_clusters_week2

        # Update the motif_cluster_mapping for the next iteration to use the new labels
        motif_cluster_mapping[week2] = updated_clusters_week2

    # Step 3: Convert the updated cluster mappings back to the array format
    for week, cluster_mapping in new_weekly_clusters.items():
        if week != 'Baseline_1':  # Skip the baseline if it was not modified
            num_motifs = len(weekly_clusters[week])
            new_weekly_clusters[week] = convert_cluster_mapping_to_array(cluster_mapping, num_motifs)
    
    return new_weekly_clusters


def convert_cluster_mapping_to_array(cluster_mapping, num_motifs):
    """
    Converts a cluster mapping to an array format where each index corresponds to a motif
    and the value is the cluster number.

    Parameters:
    cluster_mapping (dict): A dictionary where keys are cluster numbers and values are lists of motifs.
    num_motifs (int): The total number of motifs.

    Returns:
    numpy.ndarray: An array where each index corresponds to a motif and the value is the cluster number.
    """
    # Initialize an array with zeros
    cluster_array = np.zeros(num_motifs, dtype=int)

    # Fill the array with the cluster numbers
    for cluster, motifs in cluster_mapping.items():
        for motif in motifs:
            cluster_array[motif] = cluster

    return cluster_array


def find_most_similar_clusters(week1_clusters, week2_clusters):
    """
    Finds the most similar clusters between two weeks based on the motifs they contain.

    Parameters:
    week1_clusters (dict): A dictionary mapping cluster numbers to motifs for week 1.
    week2_clusters (dict): A dictionary mapping cluster numbers to motifs for week 2.

    Returns:
    dict: A dictionary mapping cluster numbers from week 1 to the most similar cluster numbers in week 2.
    """
    similarity_mapping = {}

    # Convert cluster mappings to sets for easier comparison
    week1_sets = {k: set(v) for k, v in week1_clusters.items()}
    week2_sets = {k: set(v) for k, v in week2_clusters.items()}

    # Compare each cluster in week 1 to each cluster in week 2
    for week1_cluster, week1_motifs in week1_sets.items():
        max_similarity = 0
        most_similar_cluster = None

        for week2_cluster, week2_motifs in week2_sets.items():
            # Calculate the Jaccard similarity
            intersection = len(week1_motifs & week2_motifs)
            union = len(week1_motifs | week2_motifs)
            jaccard_similarity = intersection / union if union > 0 else 0

            # Update the most similar cluster if the current one is more similar
            if jaccard_similarity > max_similarity:
                max_similarity = jaccard_similarity
                most_similar_cluster = week2_cluster

        # Map the most similar cluster from week 2 to the current cluster in week 1
        similarity_mapping[week1_cluster] = most_similar_cluster

    return similarity_mapping


def update_cluster_labels(week1_clusters, week2_clusters, similar_clusters):
    """
    Updates the cluster labels of week2 to match the most similar clusters from week1.

    Parameters:
    week1_clusters (dict): A dictionary mapping cluster numbers to motifs for week 1.
    week2_clusters (dict): A dictionary mapping cluster numbers to motifs for week 2.
    similar_clusters (dict): A dictionary mapping cluster numbers from week 1 to the most similar cluster numbers in week 2.

    Returns:
    dict: A dictionary with updated cluster labels for week 2.
    """
    # Invert the similar_clusters mapping to map week 2 clusters back to week 1
    inverted_similar_clusters = {v: k for k, v in similar_clusters.items()}

    # Create a new mapping for week 2 with updated labels
    updated_week2_clusters = {}
    for cluster, motifs in week2_clusters.items():
        # If the cluster from week 2 is similar to one in week 1, use the week 1 label
        if cluster in inverted_similar_clusters:
            new_label = inverted_similar_clusters[cluster]
            updated_week2_clusters[new_label] = motifs
        else:
            # If there's no similar cluster, keep the original label
            updated_week2_clusters[cluster] = motifs

    return updated_week2_clusters


def complete_communities_cluster(config, distance_threshold = 2, find_best_threshold = False, imagetype='.svg'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']
    
    files = AlHf.get_files(config)

    mean_transition_matrix = calculate_mean_transition_matrix(cfg, files, model_name, n_cluster)

    path_to_file = os.path.join(cfg['project_path'], 'results')

    if parameterization == 'hmm':
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters), "all_data")
    else:
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster))


    os.makedirs(aggregated_analysis_path, exist_ok=True)

    hcd_path = os.path.join(aggregated_analysis_path, "HCD")

    os.makedirs(hcd_path, exist_ok=True)

    labels = []
    motif_durations = []
    motif_frequencies = []
    avg_speed_per_motif_list = []
    transition_speed_matrices =[]
    
    for i, file in enumerate(files):
        if cfg['parameterization'] == 'hmm':
            path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster) + '-' + str(hmm_iters))
        else:
            path = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, cfg['parameterization'] + '-' + str(n_cluster))

        label_array = get_label(cfg, file, model_name, n_cluster)

        motif_duration = get_motif_duration(label_array)
        motif_frequency = get_motif_frequency(label_array)

        speed_files = glob.glob(os.path.join(path, "kinematics", f"{file}-*-speed.npy"))
        if speed_files:
            speed_data = np.load(speed_files[0])
        else:
            print(f"No speed data file found for {file} in {path}")
            continue
        
        transition_speed_matrix = calculate_average_speed_for_transitions(n_cluster, label_array, speed_data)
        avg_speed_per_motif = aggregate_speed_for_motifs(speed_data, label_array)

        motif_duration_ordered = order_dict_keys(motif_duration)
        motif_frequency_ordered = order_dict_keys(motif_frequency)
        avg_speed_per_motif_ordered = order_dict_keys(avg_speed_per_motif)
        
        labels.append(label_array)
        motif_durations.append(motif_duration_ordered)
        motif_frequencies.append(motif_frequency_ordered)
        avg_speed_per_motif_list.append(avg_speed_per_motif_ordered)
        transition_speed_matrices.append(transition_speed_matrix)


    avg_motif_duration = mean_values_of_dict_list(motif_durations)
    avg_motif_frequency = mean_values_of_dict_list(motif_frequencies)
    avg_motif_speed = mean_values_of_dict_list(avg_speed_per_motif_list)
    stacked_transition_speed_matrices = np.stack(transition_speed_matrices, axis=2)
    mean_transition_speed_matrix = np.mean(stacked_transition_speed_matrices, axis=2)

    clusters, clusters_wSpeed, normalized_features, normalized_features_wSpeed, fig = analyze_cluster_data(cfg, avg_motif_duration, avg_motif_frequency, avg_motif_speed, mean_transition_matrix, mean_transition_speed_matrix, distance_threshold)

    fig.savefig(os.path.join(aggregated_analysis_path, "HCD", f"HCD-{distance_threshold}{imagetype}"))
    plt.close(fig)
    
    fig = AlPf.plot_motif_statistics(avg_motif_duration, avg_motif_frequency, avg_motif_speed, clusters)
    fig.savefig(os.path.join(aggregated_analysis_path, f"motif_statistics_by_cluster-all_data-{distance_threshold}{imagetype}"), bbox_inches='tight')
    plt.close(fig)

    np.save(os.path.join(aggregated_analysis_path, f"avg_speed_matrix_all_data.npy"), mean_transition_speed_matrix)
    np.save(os.path.join(aggregated_analysis_path, f"avg_transition_matrix_all_data.npy"), mean_transition_matrix)
    np.save(os.path.join(aggregated_analysis_path, f"clusters_all_data-{distance_threshold}.npy"), clusters)
    np.save(os.path.join(aggregated_analysis_path, f"clusters_wSpeed_all_data-{distance_threshold}.npy"), clusters_wSpeed)


    plt.figure(figsize=(12, 12))  # Adjust the size as needed
    sns.heatmap(mean_transition_matrix, annot=True, annot_kws={"size": 3}, fmt='.4f')
    plt.xlabel("Next frame behavior", fontsize=16)
    plt.ylabel("Current frame behavior", fontsize=16)
    plt.title(f"Averaged Transition matrix of {n_cluster} clusters, All Data")
    plt.savefig(os.path.join(aggregated_analysis_path, f"avg_transition_matrix_all_data{imagetype}"), bbox_inches='tight', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 12))  # Adjust the size as needed
    sns.heatmap(mean_transition_speed_matrix, annot=True, annot_kws={"size": 3}, fmt='.2f')
    plt.xlabel("Next frame behavior", fontsize=16)
    plt.ylabel("Current frame behavior", fontsize=16)
    plt.title(f"Averaged Transition Speed matrix of {n_cluster} clusters, All Data")
    plt.savefig(os.path.join(aggregated_analysis_path, f"avg_transition_speed_matrix_all_data{imagetype}"), bbox_inches='tight', transparent=True)
    plt.close()


def group_week_files(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']

    sham_files, injured_files, treated_files, abx_files = AlHf.categorize_fileNames(config)
    file_groups = sham_files, injured_files, treated_files, abx_files
    group_names = ("Sham", "Injured", "Treated", "ABX")

    weeks = set()
    for file_group in file_groups:
        for file in file_group:
            _, study_point, _, _ = AlHf.parse_filename(file)
            weeks.add(study_point)
    
    files_by_timePoint_and_group = {}

    for week in weeks:
        path_to_file = os.path.join(cfg['project_path'], 'results')
        if parameterization == 'hmm':
            aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters))
        else:
            aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster))

        week_path = os.path.join(aggregated_analysis_path, week, )
        os.makedirs(week_path, exist_ok=True)

        files_by_timePoint_and_group[week] = {}

        for group_name, file_list in zip(group_names, file_groups):
            if week == 'Drug_Trt' and group_name == 'ABX':
                continue
            group_week_files = AlHf.get_time_point_columns_for_group(file_list, time_point=week)

            files_by_timePoint_and_group[week][group_name] = group_week_files

    return files_by_timePoint_and_group


def temp_fxn_name(config, distance_threshold = 11):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']

    path_to_file = os.path.join(cfg['project_path'], 'results')

    if parameterization == 'hmm':
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters))
    else:
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster))

    clustering = np.load(os.path.join(aggregated_analysis_path, "all_data", f"clusters_all_data-{distance_threshold}.npy"))

    # To get all of the "week" keys from the 'organized_files' dictionary and iterate over them, you can do the following:
    organized_files = group_week_files(config)
    
    for week in organized_files.keys():
        week_path = (os.path.join(aggregated_analysis_path, week))
        os.makedirs(week_path, exist_ok=True)

        # Iterate over each group within this week
        for group_name in organized_files[week]:
            if organized_files[week] == 'Drug_Trt' and group_name == 'abx':
                continue
            
            group_week_path = (os.path.join(week_path, group_name))
            os.makedirs(group_week_path, exist_ok=True)

            # Access the files for the current week and group
            files_for_week_and_group = organized_files[week][group_name]


            labels = get_labels(cfg, files_for_week_and_group, model_name, n_cluster)
            motif_usage = np.unique(labels, return_counts=True)
            motif_usage = list(motif_usage)

            group_week_trans_mat = calculate_mean_transition_matrix(cfg, files_for_week_and_group, model_name, n_cluster)
    
    
    
def temp_fxn_name_2(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']

    path_to_file = os.path.join(cfg['project_path'], 'results')

    if parameterization == 'hmm':
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters))
    else:
        aggregated_analysis_path = os.path.join(path_to_file, "aggregated_analysis", parameterization+'-'+str(n_cluster))

    clustering = np.load(os.path.join(aggregated_analysis_path, "all_data", f"clusters_all_data-{distance_threshold}.npy"))

    clustered_motifs = AlHf.group_motifs_by_cluster(clustering)

    files = AlHf.get_files(config)

    df = AlHf.combineBehavior(config, files, save=False, legacy=False)

    sham_cols, injured_cols, treated_cols, abx_cols = AlHf.categorize_columns(df)
    
    meta_data = AlHf.create_meta_data_df(df)
    
    df_long = AlHf.melt_df(df, meta_data)
    df_long = AlHf.assign_clusters(df_long, clustered_motifs)
    df_long = AlHf.assign_clusters(df_long, clustered_motifs)

    df_long_norm_base = AlHf.normalize_to_baseline(df_long)  
    df_long_norm_sham = AlHf.normalize_to_baseline_sham(df_long)
    mannUwhit_pVal_vsSham_results = ana.calculate_p_values_vs_sham(df_long_norm_sham)
    significant_results = mannUwhit_pVal_vsSham_results[mannUwhit_pVal_vsSham_results['P_Value'] < 0.05]


