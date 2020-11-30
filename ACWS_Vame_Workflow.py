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


new = True #Set to True to create new project, False to load config file
#Initialize Project:
project = 'VAME_CombinedNPW3'
directory = '/d1/studies/VAME/VAME_CombinedNPW'
modelName = 'VAME_CombinedNPW3'
file_format = '.mp4'

videoDirectory = os.path.join(directory, 'mp4s')
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

###Convert all CSVs to numpy arrays:
csvs = []
csvDirectory = os.path.join(h5Directory, 'egocentric/')
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

#Egocentric alignment:  
poseFiles = os.listdir(os.path.join(projectPath, 'videos/pose_estimation/'))
crop_size=(300,300)
for file in poseFiles:
    sampleName = file.split('-DC')[0]
    if not os.path.exists(projectPath + 'data/' + sampleName + '/' + sampleName + '-PE-seq.npy'):
        egocentric_time_series = av.align_demo(projectPath, sampleName, file_format, crop_size, use_video=False, check_video=False)
        np.save(projectPath+'data/'+sampleName+'/'+sampleName+'-PE-seq.npy', egocentric_time_series)
  
    
#Create training dataset:
vame.create_trainset(config)

#Train RNN:
vame.rnn_model(config, model_name=modelName, pretrained_weights=False, pretrained_model=None)
#Evaluate RNN:
vame.evaluate_model(config, model_name=modelName)

#Segment Behaviors:
vame.behavior_segmentation(config, model_name=modelName, cluster_method='kmeans', n_cluster=[15,30,45])

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

ctrl_mice = ['C1-RT', 'C3-RB', 'C5-NP', 'C5-RT', 'C9_LT', 'C12_NP', 'C13_RT', 'C14_LT', 'C14_LB', 'C15_RT', 'C16_RB']
cko_mice = ['C2-RB', 'C3-LT', 'C4-NP', 'C4-RT', 'C10_NP', 'C12_RT', 'C13_NP', 'C14_RT', 'C15_NP', 'C16_NP']
phases=['Saline', 'Phase1', 'Phase2', 'Phase3']

group1=ctrl_mice
group2=cko_mice
n_clusters=20


clus = [7,10,20,30]

for n_clusters in clus:
    vame.behavior_quantification(config, model_name=modelName, cluster_method='kmeans', n_cluster=n_clusters)
    hf.extractResults(projectPath, group1, group2, modelName, n_clusters, phases)

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


