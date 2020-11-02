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
from vame.analysis.videowriter import motif_videos



#Initialize Project:
project = 'VAME_Operant_NoCue'
directory = '/d1/studies/VAME/VAME_NoCue/'
videoDirectory = os.path.join(directory, 'videos')

#Initialize Project:
project = 'OperantDLC_Vame'
directory = '/d1/studies/VAME/Vame_Project/'
videoDirectory = '/d1/studies/VAME/Vame_Project/videos/'
vids = []
files = os.listdir(videoDirectory)
for f in files:
    if f.endswith(file_format):
        fullpath = os.path.join(videoDirectory, f)
        vids.append(fullpath)

if not os.path.exists(os.path.join(directory, project + '-' + creationDate + '/config.yaml')):
    config = vame.init_new_project(project=project, videos=vids, working_directory=directory, videotype='.mp4')
else:
    config = os.path.join(directory, project + '-' + creationDate + '/config.yaml')
    print("Loaded config from " + os.path.join(directory, project + '-' + creationDate + 'config.yaml'))
    
projectPath = '/'.join(config.split('/')[:-1])

###Convert h5s to egocentric CSVs:
h5Directory = os.path.join(directory, 'data/h5s')
files = os.listdir(h5Directory)
for f in files:
    if f.endswith('.h5'):
        h5Path = os.path.join(h5Directory, f)
        hf.makeEgocentricCSV_MouseCenter(h5Path, 'forepaw_r', 'forepaw_l', drop='cueLight')

###Convert all CSVs to numpy arrays:
csvs = []
csvDirectory = '/d1/studies/VAME/Vame_Project/data/'
files = os.listdir(csvDirectory)
for f in files:
    if f.endswith('.csv'):
        fullpath = os.path.join(csvDirectory, f)
        csvs.append(fullpath)

#Egocentric alignment:  
#Optional drop one of each forelimb & highlimb, keeping whichever has highest likelihood:
hf.selectLimbs(projectPath, '_7points')

#Dropping defined bodyparts:
bodyParts = ['forepaw-r', 'forepaw-l', 'hindpaw-r', 'hindpaw-l']
hf.dropBodyParts(projectPath, bodyParts)

#Egocentric alignment:
poseFiles = os.listdir(os.path.join(projectPath, 'videos/pose_estimation/'))
crop_size=(350,350)
for file in poseFiles:
    if file.endswith('.csv'):
        sampleName = file.split('-DC')[0]
        if not os.path.exists(projectPath + '/data/' + sampleName + '/' + sampleName + '-PE-seq.npy'):
            egocentric_time_series = av.alignVideo(projectPath, sampleName, file_format, crop_size, 
                                                   use_video=False, check_video=False)
            np.save(projectPath+'/data/'+sampleName+'/'+sampleName+'-PE-seq.npy', egocentric_time_series)

#Create training dataset:
vame.create_trainset(config)

#Train RNN:
vame.rnn_model(config, model_name=modelName, pretrained_weights=False, pretrained_model=None)
#Evaluate RNN:
vame.evaluate_model(config, model_name=modelName)

#Segment Behaviors:
vame.behavior_segmentation(config, model_name=modelName, cluster_method='kmeans', n_cluster=[15,30,45])
#Quantify behaviors:
vame.behavior_quantification(config, model_name=modelName, cluster_method='kmeans', n_cluster=10)
#Make Example Videos:
motif_videos(config, model_name=modelName, cluster_method="kmeans", n_cluster=[10])


#Define groups & experimental setup:
group1 = ['C1-RT', 'C3-RB', 'C5-NP', 'C5-RT', 'C9_LT', 'C12_NP', 'C13_RT', 'C14_LT', 'C14_LB', 'C15_RT', 'C16_RB']
group2 = ['C2-RB', 'C3-LT', 'C4-NP', 'C4-RT', 'C10_NP', 'C12_RT', 'C13_NP', 'C14_RT', 'C15_NP', 'C16_NP']
phases=['Saline', 'Phase1', 'Phase2', 'Phase3']

#Gather data, perform statistics, write results file:
clus=[5,10,15,20,30]
for n_clusters in clus:
    vame.behavior_quantification(config, model_name=modelName, cluster_method='kmeans', n_cluster=n_clusters)
    hf.extractResults(projectPath, group1, group2, modelName, n_clusters, phases)
                     
hf.extractResults(projectPath, group1, group2, modelName, n_clusters, phases)

from vame.analysis.videowriter import motif_videos

motif_videos(config, model_name='VAME_OperantModel2', cluster_method="kmeans", n_cluster=[10])

samples = os.listdir('/d1/studies/VAME/VAME_OperantModel2/VAME_OperantModel2-Nov4-2020/results/')
cat = pd.DataFrame()
for sample in samples:
    clu_arr = np.load('/d1/studies/VAME/VAME_OperantModel2/VAME_OperantModel2-Nov4-2020/results/' + sample + '/VAME_OperantModel2/kmeans-20/behavior_quantification/motif_usage.npy')
    clu = pd.DataFrame(clu_arr)
    clu.columns=[sample]
    cat = pd.concat([cat, clu], axis=1)
cat.to_csv('VAME_OperantModel2_10clusters_results.csv')


cat.columns

phase1 = pd.DataFrame()
phase2 = pd.DataFrame()
phase3 = pd.DataFrame()
sal = pd.DataFrame()
acute = pd.DataFrame()

for col in cat.columns:
    if col.endswith('Phase1'):
        phase1[col]=cat[col]
    if col.endswith('Phase2'):
        phase2[col]=cat[col]
    if col.endswith('Phase3'):
        phase3[col]=cat[col]
    if col.endswith('Saline'):
        sal[col]=cat[col]
    if col.endswith('5mgkg'):
        acute[col]=cat[col]

combined=pd.concat([sal, acute, phase1, phase2, phase3], axis=1)

ctrl_mice = ['C1-RT', 'C3-RB', 'C5-NP', 'C5-RT', 'C9_LT', 'C12_NP', 'C13_RT', 'C14_LT', 'C14_LB', 'C15_RT',]
cko_mice = ['C2-RB', 'C3-LT', 'C4-NP', 'C4-RT', 'C10_NP', 'C12_RT', 'C13_NP', 'C14_RT', 'C15_NP',]
ctrl=pd.DataFrame()
cko = pd.DataFrame()


for col in combined.columns:
    if col[:5] in ctrl_mice:
        ctrl[col]=combined[col]
    elif col[:6] in ctrl_mice:
        ctrl[col]=combined[col]
    elif col[:5] in cko_mice:
        cko[col]=combined[col]
    elif col[:6] in cko_mice:
        cko[col]=combined[col]
    else:
        print(str(col) + " not found in either")

ctrl['control_mean'] = np.mean(ctrl, axis=1)
ctrl['control_sem']=(np.std(ctrl, axis=1)/np.sqrt((ctrl.shape[1]-1)))
cko['cko_mean'] = np.mean(cko, axis=1)
cko['cko_sem']=(np.std(cko, axis=1)/np.sqrt((cko.shape[1]-1)))

comb = pd.concat([ctrl, cko], axis=1)
comb.to_csv(os.path.join(directory, 'CombinedResults_20Clusters.csv'))

