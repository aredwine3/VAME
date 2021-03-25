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
import matplotlib.pyplot as plt
import seaborn as sn
import vame
from vame.custom import helperFunctions as hf
from vame.custom import alignVideos as av
from vame.analysis.videowriter import motif_videos
from vame.analysis.segment_behavior import plot_transitions

#Initialize Project:
directory = '/d1/studies/VAME/VAME_VG2/'
project = 'vGluT2_RTA'
creationDate = 'Feb15-2021'
modelName = 'VG2_RTA_with6Hz'
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
vame.rnn_model(config, model_name=modelName, pretrained_weights=True, pretrained_model='VG2_RTA_with6Hz_vGluT2_RTA_Epoch203_Feb16')
#Evaluate RNN:
vame.evaluate_model(config, model_name=modelName, suffix=None)

#Segment Behaviors:
vame.behavior_segmentation(config, model_name=modelName, cluster_method='GMM', n_cluster=[9,12,15,18,20])
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
phases=['2020-12-21', '2020-12-22', '2021-01-06', '2021-01-08', '2021-01-18', '2021-01-16']
expDate = 'Dec2020'
clus=[9,12,15,18,20]
cluster_method='GMM'

rename = {'2020-12-21':'Dec2020', '2020-12-22':'Dec2020', '2021-01-06':'Jan2021', '2021-01-08':'Jan2021'}

for n_clusters in clus:
   # vame.behavior_quantification(config, model_name=modelName, cluster_method='GMM', n_cluster=n_clusters, rename=None)
  #  hf.extractResults(projectPath, expDate, group1, group2, modelName, n_clusters, phases)
    plot_transitions(config, files, n_clusters, modelName, cluster_method=cluster_method, rename=None)

    
#Make Example Videos:
#for clu in clus:
    
motif_videos(config, model_name=modelName, cluster_method="GMM", n_cluster=clus, rename=None)

files6=['VG2_RT_6Hz_2021-01-18.mp4',
 'VG1_RB_6Hz_2021-01-18.mp4',
 'VG2_LT_6Hz_2021-01-16.mp4',
 'VG3_LB_6Hz_2021-01-19.mp4',
 'VG1_RT_6Hz_2021-01-18.mp4',
 'VG1_LT_6Hz_2021-01-19.mp4',
 'VG3_RT_6Hz_2021-01-16.mp4',
 'VG1_LB_6Hz_2021-01-18.mp4']

files10=['VG2_LT_2021-01-08.mp4',
 'VG1_RB_2021-01-06.mp4',
 'VG3_RT_2021-01-08.mp4',
 'VG1_LT_2021-01-06.mp4',
 'VG2_RT_2021-01-08.mp4',
 'VG1_LB_2021-01-08.mp4']

ctrl_tms = ['VG2_RT_6Hz_2021-01-18.mp4',
 'VG3_LB_6Hz_2021-01-19.mp4',
 'VG1_LB_2021-01-08.mp4',
 'VG2_RT_2021-01-08.mp4',
 ]


vg2_tms = ['VG1_RB_6Hz_2021-01-18.mp4',
 'VG2_LT_6Hz_2021-01-16.mp4',
 'VG1_RT_6Hz_2021-01-18.mp4',
 'VG1_LT_6Hz_2021-01-19.mp4',
 'VG3_RT_6Hz_2021-01-16.mp4',
 'VG1_LB_2021-01-08.mp4']

n_cluster=20
cluster_method='GMM'
ctrl_arrays = []
for f in ctrl_tms:
    f = f.strip('.mp4')
    tm = np.load(os.path.join(projectPath, 'results/' + f + '/' + modelName + '/' + cluster_method + '-' + str(n_cluster) + '/behavior_quantification/transition_matrix.npy'))
    ctrl_arrays.append(tm)
ctrl_cat = np.stack(ctrl_arrays, axis=2)
ctrl_ave = np.mean(ctrl_cat,axis=2)

fig = plt.figure(figsize=(15,10))
fig.suptitle("Averaged Transition matrix of {} behaviors".format(tm.shape[0]))
sn.heatmap(ctrl_ave, annot=True)
plt.xlabel("Next frame behavior")
plt.ylabel("Current frame behavior")
plt.show()
fig.savefig(os.path.join(projectPath, 'Control_AverageTransitionMatrix_' + str(n_cluster) + 'clusters_GMM.png'))

vg2_arrays=[]
for f in vg2_tms:
    f = f.strip('.mp4')
    tm = np.load(os.path.join(projectPath, 'results/' + f + '/' + modelName + '/' + cluster_method + '-' + str(n_cluster) + '/behavior_quantification/transition_matrix.npy'))
    vg2_arrays.append(tm)
vg2_cat = np.stack(vg2_arrays, axis=2)
vg2_ave = np.mean(vg2_cat,axis=2)

fig = plt.figure(figsize=(15,10))
fig.suptitle("Averaged Transition matrix of {} behaviors".format(tm.shape[0]))
sn.heatmap(vg2_ave, annot=True)
plt.xlabel("Next frame behavior")
plt.ylabel("Current frame behavior")
plt.show()
fig.savefig(os.path.join(projectPath, 'ChR2_AverageTransitionMatrix_' + str(n_cluster) + 'clusters_GMM.png'))


diff = vg2_ave - ctrl_ave

fig = plt.figure(figsize=(15,10))
fig.suptitle("ChR2 - eYFP Transition matrix of {} behaviors".format(tm.shape[0]))
sn.heatmap(diff, annot=True, vmin=-.2, vmax=.2, cmap='bwr')
plt.xlabel("Next frame behavior")
plt.ylabel("Current frame behavior")
plt.show()
fig.savefig(os.path.join(projectPath, 'Difference_AverageTransitionMatrix_' + str(n_cluster) + 'clusters_GMM.png'))



