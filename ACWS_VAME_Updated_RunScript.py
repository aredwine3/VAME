#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import os
os.chdir('/d1/software/VAME')
import vame
import numpy as np
import glob
from vame.util import auxiliary as aux
from vame.analysis import behavior_structure as bs
from vame.custom import helperFunctions as hf
from vame.custom import ACWS_videowriter as avw

#%% Load old project or start new one?
new = True #set to True if you're creating a new project, leave False if resuming an old one.
    
#%% Initialize your project
# Step 1.1:
if new:
    working_directory = '/d1/studies/DLC_Data/smith/HeadTwitch_VAME'
    project='Psychadelics_SplitVideos'
    videos = ['/d1/studies/DLC_Data/smith/HeadTwitch-smith-2022-12-17/analyzedVideos']
    config = vame.init_new_project(project=project, videos=videos, working_directory=working_directory, videotype='.mp4')
elif not new:
    config = '/d1/studies/DLC_Data/smith/HeadTwitch_VAME/Psychadelics_FullDataset-Dec20-2022/config.yaml'
working_directory=os.path.dirname(config)
os.chdir(working_directory)

#%% Read paramteters from config
cfg=aux.read_config(config)
n_cluster=cfg['n_cluster']
cluster_method='kmeans'
projectPath = cfg['project_path']
modelName = cfg['model_name']
vids = cfg['video_sets']
pcutoff = cfg['pose_confidence']

#Note there is a vame.update_config(config) function that should be run after updating vame version.

#%%Optionally drop 'bodyparts' if they are stationary objects that should not be considered part of 'pose':
hf.dropBodyParts(config, ['cueLight'])

#%% Step 1.2:
# Align your behavior videos egocentric and create training dataset:
# pose_ref_index: list of reference coordinate indices for alignment
# Example: 0: snout, 1: forehand_left, 2: forehand_right, 3: hindleft, 4: hindright, 5: tail
# If you want to view the result, set check_video to True. If you want to save the result, set use_video, check_video, and save all to True.
vame.egocentric_alignment(config, crop_size=(300,300), pose_ref_index=[0,10], use_video=False, check_video=False, save=False, blank_background=False)

#%% Step 1.3:
# create the training set for the VAME model
vame.create_trainset(config)

#%% Step 2:
# Train VAME:
vame.train_model(config)

#%% Step 3:
# Evaluate model
vame.evaluate_model(config, model_name=modelName, use_snapshots=True)
hf.plotLoss(config)

#%% Step 4:
# Segment motifs/pose
vame.pose_segmentation(config)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# The following are optional choices to create motif videos, communities/hierarchies of behavior,
# community videos

#%% OPTIONIAL: Create motif videos to get insights about the fine grained poses
vame.motif_videos(config, model_name='VAME', videoType='.mp4', fps=30)

#%%Rather than the above, this line will make motif videos that contain the 
avw.motif_videos(config, extractData=True, model_name=modelName, cluster_method='hmm', videoType='.mp4', fps=30)

#%%Generate transition matrices & average transition matrixes for groups:
g1=sorted(glob.glob(os.path.join(projectPath, 'results', 'Vehicle*')))
group1 = [x.split('/')[-1] for x in g1]
g1n = 'Cue1'
g2=sorted(glob.glob(os.path.join(projectPath, 'results', 'Treatment*')))
group2 = [x.split('/')[-1] for x in g2]
g2n='Cue2'

hf.plotAverageTransitionMatrices(config, group1, group2=group2, g1name=g1n, g2name=g2n, cluster_method='hmm')

#%% Generate transition matrix and motif usage plots for each video
bs.behavior_quantification(config, modelName, cluster_method='hmm', n_cluster=15, plot=True)

#%%Generate CSVs of frames per behavior, do visual QC (check no motifs are very disproportionate between animals)
hf.combineBehavior(config, save=True, cluster_method='hmm', legacy=False)
hf.parseBehavior(config, groups=['Vehicle', 'Treatment1', 'Treatment2', 'Treatment3'], cluster_method='hmm', presession=False)

#%% OPTIONAL: Create behavioural hierarchies via community detection
hf.drawHierarchyTrees(config)
vame.community(config, show_umap=True, cut_tree=1, autofill=True)

#%% OPTIONAL: Create community videos to get insights about behavior on a hierarchical scale
vame.community_videos(config)

#%% OPTIONAL: Down projection of latent vectors and visualization via UMAP
vame.visualization(config, label='motif') #options: label: None, "motif", "community"

#%% OPTIONAL: Use the generative model (reconstruction decoder) to sample from 
# the learned data distribution, reconstruct random real samples or visualize
# the cluster center for validation
vame.generative_model(config, mode="motifs") #options: mode: "sampling", "reconstruction", "centers", "motifs"

#%% OPTIONAL: Create a video of an egocentrically aligned mouse + path through 
# the community space (similar to our gif on github) to learn more about your representation
# and have something cool to show around ;) 
# Note: This function is currently very slow. Once the frames are saved you can create a video
# or gif via e.g. ImageJ or other tools
vame.gif(config, pose_ref_index=[0,5], subtract_background=False, start=None, 
         length=500, max_lag=30, label='motif', file_format='.mp4', crop_size=(350,350))

files=["/d1/studies/DLC_Data/MouseIVSA_VAME/MouseIVSA_SmithLabVAME-Dec22-2021/IVSA_MotifUsage.csv",
"/d1/studies/DLC_Data/MouseIVSA_VAME/MouseIVSA_SmithLabVAME-Dec22-2021/Ext_MotifUsage.csv",
"/d1/studies/DLC_Data/MouseIVSA_VAME/MouseIVSA_SmithLabVAME-Dec22-2021/Cue_MotifUsage.csv"]


hf.combineMotifUsage(config, files)
