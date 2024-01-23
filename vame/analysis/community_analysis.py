#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import community as community_louvain
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.signal
import umap
from icecream import ic
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler

from vame.analysis.tree_hierarchy import (draw_tree, graph_to_tree,
                                          hierarchy_pos, traverse_tree_cutline)
from vame.util.auxiliary import read_config
from importlib import reload

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%I:%M:%S %p'
                    )


def get_adjacency_matrix(labels, n_cluster):
    temp_matrix = np.zeros((n_cluster,n_cluster), dtype=np.float64)
    adjacency_matrix = np.zeros((n_cluster,n_cluster), dtype=np.float64)
    cntMat = np.zeros((n_cluster))
    steps = len(labels)
    
    for i in range(n_cluster):
        for k in range(steps-1):
            idx = labels[k]
            if idx == i:
                idx2 = labels[k+1]
                if idx == idx2:
                    continue
                else:
                    cntMat[idx2] = cntMat[idx2] +1
        temp_matrix[i] = cntMat
        cntMat = np.zeros((n_cluster))
    
    for k in range(steps-1):
        idx = labels[k]
        idx2 = labels[k+1]
        if idx == idx2:
            continue
        adjacency_matrix[idx,idx2] = 1
        adjacency_matrix[idx2,idx] = 1
    
    transition_matrix = get_transition_matrix(temp_matrix)
    
    return adjacency_matrix, transition_matrix, temp_matrix



def get_transition_matrix_OG(adjacency_matrix, threshold = 0.0):
    adjacency_matrix = np.nan_to_num(adjacency_matrix)
    row_sum=adjacency_matrix.sum(axis=1)
    transition_matrix = adjacency_matrix/row_sum[:,np.newaxis]
    transition_matrix[transition_matrix <= threshold] = 0
    if np.any(np.isnan(transition_matrix)):
            transition_matrix=np.nan_to_num(transition_matrix)
    return transition_matrix



def get_transition_matrix(adjacency_matrix, threshold=0.0):
    row_sum = adjacency_matrix.sum(axis=1)
    # Replace zeros in row_sum with 1 to avoid division by zero
    row_sum[row_sum == 0] = 1
    transition_matrix = adjacency_matrix / row_sum[:, np.newaxis]
    transition_matrix[transition_matrix <= threshold] = 0
    # Replace NaN values with 0 if they occur after the division
    transition_matrix = np.nan_to_num(transition_matrix)
    return transition_matrix


def get_labels(cfg, files, model_name, n_cluster):
    labels = []
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    for file in files:
        if cfg['parameterization'] == 'hmm':
            path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster)+'-'+str(hmm_iters), "")
        else:
            path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data,cfg['parameterization']+'-'+str(n_cluster), "")
        label = np.load(os.path.join(path_to_file,str(n_cluster)+'_km_label_'+file+'.npy'))
        labels.append(label)
    return labels


def compute_transition_matrices(files, labels, n_cluster):
    transition_matrices = []
    for i, file in enumerate(files):
        adj, trans, mat = get_adjacency_matrix(labels[i], n_cluster)
        transition_matrices.append(trans)    
    return transition_matrices
    

def create_community_bag(files, labels, transition_matrices, cut_tree, n_cluster, autofill=False):
    # markov chain to tree -> community detection
    trees = []
    communities_all = []
    for i, file in enumerate(files):
        _, usage = np.unique(labels[i], return_counts=True)
        T = graph_to_tree(usage, transition_matrices[i], n_cluster, merge_sel=1) 
        trees.append(T)
        
        if cut_tree != None:
            community_bag =  traverse_tree_cutline(T,cutline=cut_tree, n_cluster=n_cluster, fill=autofill)
            communities_all.append(community_bag)
            draw_tree(T, file)
        else:
            draw_tree(T, file)
            flag_1 = 'no'
            plt.pause(0.5)
            while flag_1 == 'no':
                cutline = int(input("Where do you want to cut the Tree? 0/1/2/3/..."))
                community_bag =  traverse_tree_cutline(T,cutline=cutline)
                print(community_bag)
                flag_2 = input('\nAre all motifs in the list? (yes/no/restart)')
                if flag_2 == 'no':
                    while flag_2 == 'no':
                        add = input('Extend list or add in the end? (ext/end)')
                        if add == "ext":
                            motif_idx = int(input('Which motif number? '))
                            list_idx = int(input('At which position in the list? (pythonic indexing starts at 0) '))
                            community_bag[list_idx].append(motif_idx)
                        if add == "end":
                            motif_idx = int(input('Which motif number? '))
                            community_bag.append([motif_idx])
                        print(community_bag)
                        flag_2 = input('\nAre all motifs in the list? (yes/no/restart)')
                if flag_2 == 'yes':
                    flag_1 = 'yes'
                    communities_all.append(community_bag)
                if flag_2 == "restart":
                    continue     
        plt.close('all')
    return communities_all, trees


def get_community_labels(files, labels, communities_all):
    # transform kmeans parameterized latent vector into communities
    community_labels_all = []
    for k, file in enumerate(files):
        num_comm = len(communities_all[k])  
        
        community_labels = np.zeros_like(labels[k])
        for i in range(num_comm):
            clust = np.array(communities_all[k][i])
            for j in range(len(clust)):
                find_clust = np.where(labels[k] == clust[j])[0]
                community_labels[find_clust] = i
        
        community_labels = np.int64(scipy.signal.medfilt(community_labels, kernel_size=7))  
        community_labels_all.append(community_labels)

    return community_labels_all


def umap_embedding(cfg, file, model_name, n_cluster):
    reducer = umap.UMAP(n_components=2, min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], 
                        random_state=cfg['random_state']) 
    
    print("UMAP calculation for file %s" %file)
    
    if cfg['parameterization'] == 'hmm':
        folder = os.path.join(cfg['project_path'],"results",file,model_name, cfg['load_data'], cfg['parameterization']+'-'+str(n_cluster)+'-'+str(cfg['hmm_iters']),"")
    else:
        folder = os.path.join(cfg['project_path'],"results",file,model_name, cfg['load_data'], cfg['parameterization']+'-'+str(n_cluster),"")
    
    latent_vector = np.load(os.path.join(folder,'latent_vector_'+file+'.npy'))
    
    num_points = cfg['num_points']
    if num_points > latent_vector.shape[0]:
        num_points = latent_vector.shape[0]
    print("Embedding %d data points.." %num_points)
    
    embed = reducer.fit_transform(latent_vector[:num_points,:])
    
    return embed


def process_file(cfg, file, idx, model_name, load_data, parameterization, n_cluster, hmm_iters, transition_matrices, community_labels_all, communities_all, show_umap):
    if parameterization == 'hmm':
        path_to_file = os.path.join(cfg['project_path'], 'results', file, model_name, load_data, parameterization + '-' + str(n_cluster) + '-' + str(hmm_iters))
    else:
        path_to_file = os.path.join(cfg['project_path'], "results", file, model_name, load_data, parameterization + '-' + str(n_cluster), "")
    
    os.makedirs(os.path.join(path_to_file, "community"), exist_ok=True)
    
    np.save(os.path.join(path_to_file, "community", "transition_matrix_" + file + '.npy'), transition_matrices[idx])
    np.save(os.path.join(path_to_file, "community", "community_label_" + file + '.npy'), community_labels_all[idx])
    
    with open(os.path.join(path_to_file, "community", "hierarchy" + file + ".pkl"), "wb") as fp:  # Pickling
        pickle.dump(communities_all[idx], fp)
    
    if show_umap:
        embed = umap_embedding(cfg, file, model_name, n_cluster)
        umap_vis(cfg, file, embed, community_labels_all[idx], path_to_file)


# Assuming 'files' is a list of file names and other variables are defined
def process_files_in_parallel(cfg, files, model_name, load_data, parameterization, n_cluster, hmm_iters, transition_matrices, community_labels_all, communities_all, show_umap):
    # Use all available CPUs
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_file, cfg, file, idx, model_name, load_data, parameterization, n_cluster, hmm_iters, transition_matrices, community_labels_all, communities_all, show_umap) for idx, file in enumerate(files)]
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                # Get the result of the operation
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')


def umap_vis(cfg, file, embed, community_labels, path_to_file):
    """Visualize motif/community UMAPs.
    

    Parameters
    ----------
    cfg : ruamel.yaml.comments.CommentedMap (orderedDict)
        Config dictionary.
    file : string
        Path to video file to visualize UMAP of.
    embed : array-like
        Embedding created with vame.umap_embedding function.
    community_labels : array-like
        Array of community labels for each frame in file.
    path_to_file : string
        Destination to save UMAP embedding.

    Returns
    -------
    None.

    """
    num_points = cfg['num_points']
    if num_points > community_labels.shape[0]:
        num_points = community_labels.shape[0]
    print("Visualizing %d data points.." %num_points)
    
    num = np.unique(community_labels)
    
    fig = plt.figure(1)
    plt.scatter(embed[:,0], embed[:,1],  c=community_labels[:num_points], cmap='Spectral', s=2, alpha=1)
    if num.shape[0]>1:
        plt.colorbar(boundaries=np.arange(np.max(num)+2)-0.5).set_ticks(np.arange(np.max(num)+1))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    fig.savefig(os.path.join(path_to_file, file+'_UMAP.tif'))
    plt.close('all')


def community(config, show_umap=False, cut_tree=None, autofill=True):
    """Run community analysis to generate behavioral heirarchy "communities" and trees.

    Parameters
    ----------
    config : string
        Path to config.yaml file.
    show_umap : bool, optional
        Whether to show UMAP plots. The default is False.
    cut_tree : int, optional
        Level at which to cut the heirarchy tree, see main VAME docs. The default is None, which prompts you to input where to cut each tree individually.
    autofill : bool, optional
        Whether to fill any unfound communities with 0. The default is True, setting to False leads to error if any communities are found with 0 frames in any video.

    Returns
    -------
    None.

    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    load_data = cfg['load_data']
    hmm_iters = cfg['hmm_iters']
    parameterization = cfg['parameterization']
    
    files = []
    if cfg['all_data'] == 'No' or cfg['all_data']=='no':
        all_flag = input("Do you want to write motif videos for your entire dataset? \n"
                     "If you only want to use a specific dataset type filename: \n"
                     "yes/no/filename ")
    else:
        all_flag = 'yes'

    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)

    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        files.append(all_flag)
    
    labels = get_labels(cfg, files, model_name, n_cluster)
    transition_matrices = compute_transition_matrices(files, labels, n_cluster)
    communities_all, trees = create_community_bag(files, labels, transition_matrices, cut_tree, n_cluster, autofill=autofill)
    community_labels_all = get_community_labels(files, labels, communities_all)    
    
    if os.cpu_count() > 1:
        process_files_in_parallel(cfg, files, model_name, load_data, parameterization, n_cluster, hmm_iters, transition_matrices, community_labels_all, communities_all, show_umap)
    else:
        for idx, file in enumerate(files):
            if parameterization == 'hmm':
                path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,load_data, parameterization +'-'+str(n_cluster)+'-'+str(hmm_iters))
            else:
                path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,load_data, parameterization +'-'+str(n_cluster),"")
            
            if not os.path.exists(os.path.join(path_to_file,"community")):
                os.mkdir(os.path.join(path_to_file,"community"))
            
            np.save(os.path.join(path_to_file,"community","transition_matrix_"+file+'.npy'),transition_matrices[idx])
            np.save(os.path.join(path_to_file,"community","community_label_"+file+'.npy'), community_labels_all[idx])
            
            with open(os.path.join(path_to_file,"community","hierarchy"+file+".pkl"), "wb") as fp:   #Pickling
                pickle.dump(communities_all[idx], fp)
        
            if show_umap == True:
                embed = umap_embedding(cfg, file, model_name, n_cluster)
                umap_vis(cfg, file, embed, community_labels_all[idx], path_to_file)
    
# with open(os.path.join(path_to_file,"community","","hierarchy"+file+".txt"), "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
