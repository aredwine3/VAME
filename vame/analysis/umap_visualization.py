#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import glob
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from vame.custom.ALR_helperFunctions import get_files
from vame.util.auxiliary import read_config

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)å
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server


def umap_vis(file, embed, num_points):        
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1], s=2, alpha=.5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    plt.close('all')

def umap_label_vis(file, embed, label, n_cluster, num_points, path_to_file):
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1],  c=label[:num_points], cmap='Spectral', s=2, alpha=.7)
    plt.colorbar(boundaries=np.arange(n_cluster+1)-0.5).set_ticks(np.arange(n_cluster))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    fig.savefig(os.path.join(path_to_file, file+'UMAP_LabeledMotifs.png'))
    plt.close('all')

def umap_vis_comm(file, embed, community_label, num_points, path_to_file):
    num = np.unique(community_label).shape[0]
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1],  c=community_label[:num_points], cmap='Spectral', s=2, alpha=.7)
    plt.colorbar(boundaries=np.arange(num+1)-0.5).set_ticks(np.arange(num))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    fig.savefig(os.path.join(path_to_file, file+'UMAP_LabeledCommunities.png'))    
    plt.close('all')

def visualization(config, label=None):
    config_file = Path(config).resolve()
    cfg = read_config(config_file) 
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    hmm_iters = cfg['hmm_iters']
    load_data = cfg['load_data']

    files = get_files(config)

    for file in files:
        path_to_file=os.path.join(cfg['project_path'],"results",file,"",model_name,load_data,'kmeans-'+str(n_cluster))
        print("Constructed path to file:", path_to_file)
        
        try:
            embed = np.load(os.path.join(path_to_file,"","community","","umap_embedding_"+file+".npy"))
            num_points = cfg['num_points']
            if num_points > embed.shape[0]:
                num_points = embed.shape[0]
        except:
            if not os.path.exists(os.path.join(path_to_file,"community")):
                os.mkdir(os.path.join(path_to_file,"community"))
            print("Compute embedding for file %s" %file)
            reducer = umap.UMAP(n_components=2, min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], 
                    random_state=cfg['random_state']) 
            
            latent_vector = np.load(os.path.join(path_to_file,"",'latent_vector_'+file+'.npy'))
            
            num_points = cfg['num_points']
            if num_points > latent_vector.shape[0]:
                num_points = latent_vector.shape[0]
            print("Embedding %d data points.." %num_points)
            
            embed = reducer.fit_transform(latent_vector[:num_points,:])
            np.save(os.path.join(path_to_file,"community","umap_embedding_"+file+'.npy'), embed)
        
        print("Visualizing %d data points.. " %num_points)
        if label == None:                    
            umap_vis(file, embed, num_points)
            
        if label == 'motif':
            if len(glob.glob(os.path.join(path_to_file,'*LabeledMotifs.png')))<1:
                motif_label = np.load(os.path.join(path_to_file,str(n_cluster)+'_km_label_'+file+'.npy'))
                umap_label_vis(file, embed, motif_label, n_cluster, num_points, path_to_file)
            else:
                print("Motif UMAP for " + file + " already found, skipping...")

        if label == "community":
            if len(glob.glob(os.path.join(path_to_file,'*LabeledCommunities.png')))<1:
                community_label = np.load(os.path.join(path_to_file,"community","community_label_"+file+".npy"))
                umap_vis_comm(file, embed, community_label, num_points, path_to_file)                                    
            else:
                print("Community UMAP for " + file + " already found, skipping...")


def process_umaps(config, label=None):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    files = get_files(config)  # Make sure get_files returns a list of file names

    # Create a partial function that has the config parameter filled
    visualize_with_config = partial(visualize_single_file, config)

    # Use all available CPU cores
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(visualize_with_config, files)

def visualize_single_file(config, file, label='motif'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file) 
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    hmm_iters = cfg['hmm_iters']
    load_data = cfg['load_data']


    path_to_file = os.path.join(cfg['project_path'], "results", file, model_name, load_data, 'kmeans-' + str(n_cluster))
    community_path = os.path.join(path_to_file, "community")
    embedding_file = os.path.join(community_path, f"umap_embedding_{file}.npy")
    latent_vector_file = os.path.join(path_to_file, f'latent_vector_{file}.npy')

    if not os.path.exists(embedding_file):
        print(f"Embedding file not found: {embedding_file}")
    if not os.path.exists(latent_vector_file):
        print(f"Latent vector file not found: {latent_vector_file}")
    
    try:
        embed = np.load(embedding_file)
        num_points = cfg['num_points']
        if num_points > embed.shape[0]:
            num_points = embed.shape[0]
    except FileNotFoundError:
        os.makedirs(community_path, exist_ok=True)
        print("Compute embedding for file %s" %file)
        reducer = umap.UMAP(n_components=2, min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], 
                random_state=cfg['random_state']) 
        
        
        latent_vector = np.load(os.path.join(path_to_file,"",'latent_vector_'+file+'.npy'))
        
        num_points = cfg['num_points']
        if num_points > latent_vector.shape[0]:
            num_points = latent_vector.shape[0]
        print("Embedding %d data points.." %num_points)
        
        embed = reducer.fit_transform(latent_vector[:num_points,:])
        np.save(os.path.join(path_to_file,"community","umap_embedding_"+file+'.npy'), embed)
    
    print("Visualizing %d data points.. " %num_points)
    if label == None:                    
        umap_vis(file, embed, num_points)
        
    if label == 'motif':
        if len(glob.glob(os.path.join(path_to_file,'*LabeledMotifs.png')))<1:
            motif_label = np.load(os.path.join(path_to_file,str(n_cluster)+'_km_label_'+file+'.npy'))
            umap_label_vis(file, embed, motif_label, n_cluster, num_points, path_to_file)
        else:
            print("Motif UMAP for " + file + " already found, skipping...")

    if label == "community":
        if len(glob.glob(os.path.join(path_to_file,'*LabeledCommunities.png')))<1:
            community_label = np.load(os.path.join(path_to_file,"community","community_label_"+file+".npy"))
            umap_vis_comm(file, embed, community_label, num_points, path_to_file)                                    
        else:
            print("Community UMAP for " + file + " already found, skipping...")
