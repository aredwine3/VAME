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
from vame.analysis.segment_behavior import plot_transitions

#Initialize Project:
directory = '/d1/studies/VAME/VAME_VG2/'
project = 'vGluT2_RTA'
creationDate = 'Feb6-2021'
modelName = 'VG2_RTA6'
file_format = '.mp4'

videoDirectory = '/d1/studies/VAME/VAME_VG2/vGluT2_RTA-Jan8-2021/videos/'
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

#Selecting Limbs:
#hf.selectLimbs(projectPath, '_7points')
hf.dropBodyParts(projectPath, ['LED'])

#Egocentric alignment:
poseFiles = os.listdir(os.path.join(projectPath, 'videos/pose_estimation/'))
crop_size=(375,375)
for file in poseFiles:
    if file.endswith('.csv'):
        sampleName = file.split('-DC')[0]
        if not os.path.exists(projectPath + '/data/' + sampleName + '/' + sampleName + '-PE-seq.npy'):
            egocentric_time_series = av.alignVideo(projectPath, sampleName, file_format, crop_size, use_video=False, check_video=False, save=False)
            df = pd.DataFrame(egocentric_time_series).T
            np.save(projectPath+'/data/'+sampleName+'/'+sampleName+'-PE-seq.npy', egocentric_time_series)
            if not os.path.exists(os.path.join(projectPath, 'videos/pose_estimation/egocentric/')):
                os.mkdir(os.path.join(projectPath, 'videos/pose_estimation/egocentric/'))
            df.to_csv(os.path.join(projectPath, 'videos/pose_estimation/egocentric/' + sampleName + '_egocentric.csv'))

    
    
#Create training dataset:
vame.create_trainset(config)

#Train RNN:
vame.rnn_model(config, model_name=modelName, pretrained_weights=False, pretrained_model=None)
#Evaluate RNN:
vame.evaluate_model(config, model_name=modelName, suffix=None)

#Segment Behaviors:
vame.behavior_segmentation(config, model_name=modelName, cluster_method='kmeans', n_cluster=[10,15,20,25,30])
#Quantify behaviors:
vame.behavior_quantification(config, model_name=modelName, cluster_method='ts-kmeans', n_cluster=30)
#Plot transition matrices
files = os.listdir(os.path.join(projectPath, 'results/'))
n_cluster=10
plot_transitions(config, files, n_cluster, modelName, cluster_method='kmeans')

###Run groupwise comparisons
ctrl_mice = ['C1-RT', 'C3-RB', 'C5-NP', 'C5-RT', 'C9_LT', 'C12_NP', 'C13_RT', 'C14_LT', 'C14_LB', 'C15_RT', 'C16_RB']
cko_mice = ['C2-RB', 'C3-LT', 'C4-NP', 'C4-RT', 'C10_NP', 'C12_RT', 'C13_NP', 'C14_RT', 'C15_NP', 'C16_NP']
#phases=['Saline', 'Phase1', 'Phase2', 'Phase3']
phases = ['11-06', '10-24']
group1=ctrl_mice
group2=cko_mice



group1 = ['VG1_RT', 'VG1_RB', 'VG1_LT', 'VG2_LT', 'VG3_RT']
group2 = ['VG1_LB', 'VG2_RT', 'VG3_LB']
#phases=['2020-12-21', '2020-12-22', '2021-01-06', '2021-01-08']
phases=['Dec2020', 'Jan2021']

"VG2_RT_2020-12-21",
"VG1_RB_2020-12-21",
"VG1_LB_2021-01-08",
"VG3_RT_2021-01-08",
"VG3_RT_2020-12-22",
"VG3_LB_2020-12-22",
"VG2_RT_2021-01-08",
"VG2_LT_2021-01-08",
"VG2_LT_2020-12-22",
"VG1_RT_2021-01-06",
"VG1_RT_2020-12-21",
"VG1_RB_2021-01-06",
"VG1_LT_2021-01-06",
"VG1_LT_2020-12-21",
"VG1_LB_2020-12-21"

expDate = 'Dec2020'
clus=[10,15,20,25,30]

rename = {'2020-12-21':'Dec2020', '2020-12-22':'Dec2020', '2021-01-06':'Jan2021', '2021-01-08':'Jan2021'}

for n_clusters in clus:
    vame.behavior_quantification(config, model_name=modelName, cluster_method='kmeans', n_cluster=n_clusters, rename=rename)
    hf.extractResults(projectPath, expDate, group1, group2, modelName, n_clusters, phases)
    plot_transitions(config, files, n_clusters, modelName, cluster_method='kmeans', rename=rename)

    
#Make Example Videos:
motif_videos(config, model_name=modelName, cluster_method="kmeans", n_cluster=[10,15,20,25])



