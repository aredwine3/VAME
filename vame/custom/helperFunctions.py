#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:52:04 2020

@author: luxemk
"""

import numpy as np
import pandas as pd
import os
os.chdir('/d1/studies/VAME')
from pathlib import Path
from vame.util.auxiliary import read_config



def makeEgocentricCSV(h5Path, bodyPart):
    directory = '/'.join(h5Path.split('/')[:-1])
    fileName = h5Path.split('/')[-1]#.split('DLC')[0]
    f, e = os.path.splitext(fileName)
    df = pd.read_hdf(h5Path)
    cols = df.columns
    newCols = cols.droplevel(level=0) #drops the 'scorer' level from the DLC data MultiIndex
    df.columns = newCols
    bodyParts = []
    for col in newCols:
        bp = col[0]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts)) #remove duplicates
    df_ego = df
    bodyParts_norm = bodyParts
    bodyParts_norm.remove(bodyPart)
    for bp in bodyParts_norm: #normalize bodyparts by subtracting one from all 
        df_ego[(bp, 'x')] = df_ego[(bp, 'x')] - df_ego[(bodyPart, 'x')]
        df_ego[(bp, 'y')] = df_ego[(bp, 'y')] - df_ego[(bodyPart, 'y')]
    df_ego[(bodyPart, 'x')] = df_ego[(bodyPart, 'x')] - df_ego[(bodyPart, 'x')] 
    df_ego[(bodyPart, 'y')] = df_ego[(bodyPart, 'y')] - df_ego[(bodyPart, 'y')]
    if not os.path.exists(os.path.join(directory, 'egocentric/')):
        os.mkdir(os.path.join(directory, 'egocentric/'))
    df_ego.to_csv(os.path.join(directory, 'egocentric/' + f + '_egocentric.csv'))

def makeEgocentricCSV_Center(h5Path, bodyPart1, bodyPart2, drop=None):
    directory = '/'.join(h5Path.split('/')[:-1])
    fileName = h5Path.split('/')[-1]#.split('DLC')[0]
    f, e = os.path.splitext(fileName)
    df = pd.read_hdf(h5Path)
    cols = df.columns
    newCols = cols.droplevel(level=0) #drops the 'scorer' level from the DLC data MultiIndex
    df.columns = newCols
    if drop: #if you want to drop a bodypart from the data you can use this argument
        df.drop(labels=drop, axis=1, level=0, inplace=True)
    bodyParts = []
    cols = df.columns
    for col in cols:
        bp = col[0]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts)) #remove duplicates
    df_ego = df
    a_x = pd.DataFrame([df_ego[(bodyPart1, 'x')], df_ego[(bodyPart2, 'x')]]).T
    a_y = pd.DataFrame([df_ego[(bodyPart1, 'y')], df_ego[(bodyPart2, 'y')]]).T
    df_ego[('egoCentricPoint', 'x')] = np.mean(a_x, axis=1)
    df_ego[('egoCentricPoint', 'y')] = np.mean(a_y, axis=1)
    for bp in bodyParts: #normalize to mean x/y of two specified bodyparts
        df_ego[(bp, 'x')] = df_ego[(bp, 'x')] - df_ego[('egoCentricPoint', 'x')]
        df_ego[(bp, 'y')] = df_ego[(bp, 'y')] - df_ego[('egoCentricPoint', 'y')]
    if not os.path.exists(os.path.join(directory, 'egocentric/')):
        os.mkdir(os.path.join(directory, 'egocentric/'))
    df_ego.to_csv(os.path.join(directory, 'egocentric/' + f + '_egocentric_centered.csv'))

    
def csv_to_numpy(projectPath, csvPath, pcutoff=.99):
    """
    This is a demo function to show how a conversion from the resulting pose-estimation.csv file
    to a numpy array can be implemented. 
    Note that this code is only useful for data which is a priori egocentric, i.e. head-fixed
    or otherwise restrained animals. 
    """
    fileName = csvPath.split('/')[-1].split('DLC')[0]
    f, e = os.path.splitext(fileName)
    # Read in your .csv file, skip the first two rows and create a numpy array
    data = pd.read_csv(csvPath, skiprows = 1)
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:,1:] 
    
    # get the number of bodyparts, their x,y-position and the confidence from DeepLabCut
    bodyparts = int(np.size(data_mat[0,:]) / 3)
    positions = []
    confidence = []
    idx = 0
    for i in range(bodyparts):
        positions.append(data_mat[:,idx:idx+2])
        confidence.append(data_mat[:,idx+2])
        idx += 3
    
    body_position = np.concatenate(positions, axis=1)
    con_arr = np.array(confidence)
    
    # find low confidence and set them to NaN (vame.create_trainset(config) will interpolate these NaNs)
    body_position_nan = []
    idx = -1
    for i in range(bodyparts*2):
        if i % 2 == 0:
            idx +=1
        seq = body_position[:,i]
        seq[con_arr[idx,:]<pcutoff] = np.NaN
        body_position_nan.append(seq)
    
    final_positions = np.array(body_position_nan)
    # save the final_positions array with np.save()
    np.save(os.path.join(projectPath, 'data/' + f + '/' + f + "-PE-seq.npy"), final_positions)

def combineBehavior(config, save=True, n_cluster=30):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    project_path = cfg['project_path']
    files = []
    if cfg['all_data'] == 'No':
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
    
    cat = pd.DataFrame()
    for file in files:
        arr = np.load(os.path.join(project_path, 'results/' + file + '/VAME_NPW/kmeans-' + str(n_cluster) + '/behavior_quantification/motif_usage.npy'))
        df = pd.DataFrame(arr, columns=[file])
        cat = pd.concat([cat, df], axis=1)

    phases=[]
    mice=[]
    for col in cat.columns:
        phase = col.split('_')[1]
        phases.append(phase)
        mouse = col.split('_')[0]
        mice.append(mouse)
    lz = list(zip(mice, phases))    
    lz = sorted(lz)
    joined = []
    for pair in lz:
        j = '_'.join(pair)
        joined.append(j)
    ind = sorted(lz)
    ind = pd.MultiIndex.from_tuples(ind)
    df2 = pd.DataFrame()
    for sample in joined:
        df2[sample] = cat[sample]
    df2.columns=ind
    
    if save:
        df2.to_csv(os.path.join(project_path, 'results/Motif_Usage_Combined_' + str(n_cluster) + 'clusters.csv'))
    return(df2)

