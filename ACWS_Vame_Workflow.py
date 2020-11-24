#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:46:40 2020

@author: smith
"""

import os
import numpy as np
import pandas as pd
os.chdir('/d1/studies/VAME/')
import vame
from vame.custom import helperFunctions as hf
from vame.custom import alignVideos as av


new = False #Set to True to create new project, False to load config file

#Initialize Project:
project = 'VAME_CombinedNPW_Zdim20'
directory = '/d1/studies/VAME/VAME_CombinedNPW'
modelName = 'VAME_CombinedNPW_Zdim20'
videoDirectory = os.path.join(directory, 'mp4s')
vids = []
files = os.listdir(videoDirectory)
for f in files:
    if f.endswith('.mp4'):
        fullpath = os.path.join(videoDirectory, f)
        vids.append(fullpath)
        
if new:
    config = vame.init_new_project(project=project, videos=vids, working_directory=directory, videotype='.mp4')


config = '/d1/studies/VAME/VAME_CombinedNPW/VAME_CombinedNPW-Nov11-2020/config.yaml'

projectPath = '/'.join(config.split('/')[:-1])
    
    
>>>>>>> Added workflow
###Convert h5s to egocentric CSVs:
h5Directory = os.path.join(directory, 'h5s')
files = os.listdir(h5Directory)
for f in files:
    if f.endswith('.h5'):
        h5Path = os.path.join(h5Directory, f)
        hf.makeEgocentricCSV_Center(h5Path, 'nose', 'tail-base', drop=None)


###Convert all CSVs to numpy arrays:
csvs = []
csvDirectory = os.path.join(h5Directory, 'egocentric/')
files = os.listdir(csvDirectory)
for f in files:
    if f.endswith('.csv'):
        fullpath = os.path.join(csvDirectory, f)
        csvs.append(fullpath)

for f in csvs:
    hf.csv_to_numpy(projectPath, f, pcutoff=0.9)

#Create training dataset:
vame.create_trainset(config)

#Train RNN:
vame.rnn_model(config, model_name=modelName, pretrained_weights=False, pretrained_model=None)
#Evaluate RNN:
vame.evaluate_model(config, model_name=modelName)

#Segment Behaviors:
vame.behavior_segmentation(config, model_name=modelName, cluster_method='kmeans', n_cluster=[10,20,30])
#Quantify behaviors:
vame.behavior_quantification(config, model_name=modelName, cluster_method='kmeans', n_cluster=30)

from vame.analysis.videowriter import motif_videos

motif_videos(config, model_name=modelName, cluster_method="kmeans", n_cluster=[30])

samples = os.listdir(os.path.join(projectPath, 'results/'))
cat = pd.DataFrame()
for sample in samples:
    clu_arr = np.load(os.path.join(projectPath, 'results/' + sample + '/VAME_CombinedNPW_NewAlignment/kmeans-30/behavior_quantification/motif_usage.npy'))
    clu = pd.DataFrame(clu_arr)
    clu.columns=[sample]
    cat = pd.concat([cat, clu], axis=1)
cat.to_csv(os.path.join(directory, 'VAME_NPW_NewAlignment_Test_30clusters_results.csv'))


cat.columns

ctrl_mice = ['C1-RT', 'C3-RB', 'C5-NP', 'C5-RT', 'C9_LT', 'C12_NP', 'C13_RT', 'C14_LT', 'C14_LB', 'C15_RT', 'C16_RB']
cko_mice = ['C2-RB', 'C3-LT', 'C4-NP', 'C4-RT', 'C10_NP', 'C12_RT', 'C13_NP', 'C14_RT', 'C15_NP', 'C16_NP']
ctrl=pd.DataFrame()
cko = pd.DataFrame()



#ctrl_mice = ['C9_LT_', 'C12_NP', 'C15_RT', 'C16_NP']
#cko_mice = ['C10_NP', 'C15_NP', 'C16_RB']


for col in cat.columns:
    if col[:6] in ctrl_mice:
        ctrl[col]=cat[col]
    elif col[:5] in ctrl_mice:
        ctrl[col]=cat[col]
    elif col[:6] in cko_mice:
        cko[col]=cat[col]
    elif col[:5] in cko_mice:
        cko[col]=cat[col]
    else:
        print(str(col) + " not found in either")

ctrl['control_mean'] = np.mean(ctrl, axis=1)
ctrl['control_sem']=(np.std(ctrl, axis=1)/np.sqrt((ctrl.shape[1]-1)))
cko['cko_mean'] = np.mean(cko, axis=1)
cko['cko_sem']=(np.std(cko, axis=1)/np.sqrt((cko.shape[1]-1)))

ctrl.to_csv(os.path.join(directory, 'Control_CombinedNPW_NewAlignment_30Clusters.csv'))
cko.to_csv(os.path.join(directory, 'cKO_CombinedNPW_NewAlignment_30Clusters.csv'))
comb = pd.concat([ctrl, cko], axis=1)
comb.to_csv(os.path.join(directory, 'CombinedResults_NewAlignment_30Clusters.csv'))

cols = list(ctrl.columns)
for col in cols:
    if col.endswith('_2020-11-09'):
        newCol = '_'.join(col.split('_')[:-1])
        i = cols.index(col)
        cols.remove(col)
        cols.insert(i, newCol)
ctrl.columns=cols

phases=['Phase1', 'Phase2', 'Phase3']
ctrl_p1 = pd.DataFrame()
ctrl_p2 = pd.DataFrame()
ctrl_p3 = pd.DataFrame()
cko_p1 = pd.DataFrame()
cko_p2 = pd.DataFrame()
cko_p3 = pd.DataFrame()
for col in ctrl.columns:
    if col.endswith('Phase2'):
        ctrl_p2[col]=ctrl[col]

for col in cko.columns:
    if col.endswith('Phase2'):
        cko_p2[col]=cko[col]

ctrl_p2.to_csv(os.path.join(directory, 'Ctrl_NewAlignment_Phase2_30clusters.csv'))
cko_p2.to_csv(os.path.join(directory, 'cKO_NewAlignment_Phase2_30clusters.csv'))

results = pd.concat([ctrl_p1, cko_p1], axis=1)
results.to_csv(os.path.join(directory, 'Combined_NewAlignment_Phase2_30clusters_Results.csv'))

