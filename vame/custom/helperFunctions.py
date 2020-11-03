#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:52:04 2020

@author: luxemk
"""

import numpy as np
import pandas as pd
import os

def makeEgocentricCSV(h5Path, bodyPart):
    directory = '/'.join(h5Path.split('/')[:-1])
    fileName = h5Path.split('/')[-1]#.split('DLC')[0]
    f, e = os.path.splitext(fileName)
    df = pd.read_hdf(h5Path)
    cols = df.columns
    newCols = cols.droplevel(level=0)
    df.columns = newCols
    bodyParts = []
    for col in newCols:
        bp = col[0]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts))
#    for bp in bodyParts:
#        df.drop(labels=[(bp, 'likelihood')], axis=1, inplace=True)
    df_ego = df
    bodyParts_norm = bodyParts
    bodyParts_norm.remove(bodyPart)
    for bp in bodyParts_norm:
        df_ego[(bp, 'x')] = df_ego[(bp, 'x')] - df_ego[(bodyPart, 'x')]
        df_ego[(bp, 'y')] = df_ego[(bp, 'y')] - df_ego[(bodyPart, 'y')]
    df_ego[(bodyPart, 'x')] = df_ego[(bodyPart, 'x')] - df_ego[(bodyPart, 'x')] 
    df_ego[(bodyPart, 'y')] = df_ego[(bodyPart, 'y')] - df_ego[(bodyPart, 'y')]
    if not os.path.exists(os.path.join(directory, 'egocentric/')):
        os.mkdir(os.path.join(directory, 'egocentric/'))
    df_ego.to_csv(os.path.join(directory, 'egocentric/' + f + '_egocentric.csv'))
    
    
def csv_to_numpy(projectPath, csvPath):
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
        seq[con_arr[idx,:]<.99] = np.NaN
        body_position_nan.append(seq)
    
    final_positions = np.array(body_position_nan)
    
    # save the final_positions array with np.save()
    np.save(os.path.join(projectPath, 'data/' + f + '/' + f + "-PE-seq.npy"), final_positions)

