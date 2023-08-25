#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import numpy as np

from pathlib import Path
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

from vame.util.auxiliary import read_config
import glob

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
    
    return adjacency_matrix, transition_matrix


def get_transition_matrix(adjacency_matrix, threshold = 0.0):
    row_sum=adjacency_matrix.sum(axis=1)
    transition_matrix = adjacency_matrix/row_sum[:,np.newaxis]
    transition_matrix[transition_matrix <= threshold] = 0
    if np.any(np.isnan(transition_matrix)):
            transition_matrix=np.nan_to_num(transition_matrix)
    return transition_matrix

    
def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
    
def get_network(path_to_file, file, cluster_method, n_cluster, plot=False, imagetype='.pdf'):
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
    if cluster_method == 'kmeans' or cluster_method=='hmm':
        labels = np.load(path_to_file + '/'+str(n_cluster)+'_km_label_'+file+'.npy')
    elif cluster_method=='ts-kmeans':
        labels = np.load(path_to_file + '/'+str(n_cluster)+'_ts-kmeans_label_'+file+'.npy')
    else:
        labels = np.load(path_to_file + '/'+str(n_cluster)+'_gmm_label_'+file+'.npy')
        
    adj_mat, transition_matrix = get_adjacency_matrix(labels, n_cluster=n_cluster)       
    motif_usage = np.unique(labels, return_counts=True)
    cons = consecutive(motif_usage[0])
    if len(cons) != 1:
        used_motifs = list(motif_usage[0])
        usage_list = list(motif_usage[1])
        
        for i in range(n_cluster):
            if i not in used_motifs:
                used_motifs.insert(i, i)
                usage_list.insert(i,0)
        
        usage = np.array(usage_list)
    
        motif_usage = usage
    else:
        motif_usage = motif_usage[1]
    
    np.save(os.path.join(path_to_file, 'behavior_quantification', 'adjacency_matrix.npy'), adj_mat)
    np.save(os.path.join(path_to_file, 'behavior_quantification', 'transition_matrix.npy'), transition_matrix)
    np.save(os.path.join(path_to_file, 'behavior_quantification', 'motif_usage.npy'), motif_usage)
    
    if plot:
        tmplot = sns.heatmap(transition_matrix, annot=True)
        plt.xlabel("Next frame behavior", fontsize=16)
        plt.ylabel("Current frame behavior", fontsize=16)
        plt.title("Averaged Transition matrix of " + str(n_cluster) + " clusters")
        plt.savefig(os.path.join(path_to_file, 'behavior_quantification', file+'_transition_matrix'+imagetype), bbox_inches='tight', transparent=imagetype=='.pdf')
        plt.close()
        
        motplot = sns.barplot(x=list(range(n_cluster)), y=motif_usage)
        plt.xlabel('Cluster ID', fontsize=16)
        plt.ylabel('Number of frames in cluster', fontsize=16)
        plt.savefig(os.path.join(path_to_file, 'behavior_quantification', file+'_motif_usage'+imagetype), bbox_inches='tight', transparent=imagetype=='.pdf')
    
def behavior_quantification(config, model_name, cluster_method='kmeans', n_cluster=30, plot=False, rename=False, imagetype='.pdf'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    files = []
    if cfg['all_data'] == 'No':
        all_flag = input("Do you want to quantify your entire dataset? \n"
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
        
    if rename:
        count=len(files)
        for f in files[:count]:
            suffix=f.split('_')[-1]
            fn = f.replace(suffix, rename[suffix])
            files.append(fn)         
        files = files[count:]
        
    for file in files:
        path_to_file=os.path.join(cfg['project_path'],'results',file,model_name,cluster_method+'-'+str(n_cluster))
       
        if not os.path.exists(os.path.join(path_to_file,'behavior_quantification')):
            os.mkdir(os.path.join(path_to_file,'behavior_quantification'))
        if rename:
            for filename in glob.iglob(path_to_file+'/'+str(n_cluster) + '*.npy'):
                file, ext = os.path.splitext(filename.split('/')[-1])
                file = file.split(str(n_cluster)+'_km_label_')[-1]
        get_network(path_to_file, file, cluster_method, n_cluster, plot=plot, imagetype=imagetype)

        






   

