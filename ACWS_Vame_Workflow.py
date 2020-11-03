#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:46:40 2020

@author: smith
"""

import os
import numpy as np
import pandas as pd
import vame
from vame.custom import helperFunctions as hf

new = False
#Initialize Project:
project = 'OperantDLC_Vame'
directory = '/d1/studies/VAME/Vame_Project/'
videoDirectory = '/d1/studies/VAME/Vame_Project/videos/'
vids = []
files = os.listdir(videoDirectory)
for f in files:
    if f.endswith('.mp4'):
        fullpath = os.path.join(videoDirectory, f)
        vids.append(fullpath)
        
if new:
    config = vame.init_new_project(project=project, videos=vids, working_directory=directory, videotype='.mp4')
elif not new:
    config = '/d1/studies/VAME/Vame_Project/OperantDLC_Vame-Nov2-2020/config.yaml'
    projectPath = '/'.join(config.split('/')[:-1])

###Convert h5s to egocentric CSVs:
h5Directory = '/d1/studies/VAME/Vame_Project/data/h5s/'
files = os.listdir(h5Directory)
for f in files:
    if f.endswith('.h5'):
        h5Path = os.path.join(h5Directory, f)
        hf.makeEgocentricCSV(h5Path, 'nose')


###Convert all CSVs to numpy arrays:
csvs = []
csvDirectory = '/d1/studies/VAME/Vame_Project/data/h5s/egocentric/'
files = os.listdir(csvDirectory)
for f in files:
    if f.endswith('.csv'):
        fullpath = os.path.join(csvDirectory, f)
        csvs.append(fullpath)

for f in csvs:
    hf.csv_to_numpy(projectPath, f)

#Create training dataset:
vame.create_trainset(config)

#Train RNN:
vame.rnn_model(config, model_name='VAME_OperantDLC', pretrained_weights=False, pretrained_model=None)
#Evaluate RNN:
vame.evaluate_model(config, model_name='VAME_OperantDLC')

#Segment Behaviors:
vame.behavior_segmentation(config, model_name='VAME_OperantDLC', cluster_method='kmeans', n_cluster=10)








