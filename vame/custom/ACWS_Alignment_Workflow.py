#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:30:24 2020

@author: smith
"""

import os

import matplotlib

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server
    
os.chdir('/d1/studies/VAME/')
import cv2 as cv
import numpy as np
import pandas as pd
import tqdm      
from vame.custom import alignVideos as av


path_to_file = "/d1/studies/VAME/VAME_CombinedNPW/VAME_CombinedNPW3-Nov24-2020/"
filename = 'C16_RB_Phase2_2020-11-09'
file_format='.mp4'
crop_size=(300,300)

# call function and save into your VAME data folder
egocentric_time_series = av.align_demo(path_to_file, filename, file_format, crop_size, use_video=True, check_video=True)
np.save(path_to_file+'data/'+filename+'/'+filename+'-PE-seq.npy', egocentric_time_series)

# test plot
import matplotlib.pyplot as plt
plt.plot(egocentric_time_series.T)
plt.savefig(os.path.join(path_to_file, 'data/' + filename + '/' + filename + '_aligned.tif'))


poseFiles = os.listdir('/d1/studies/VAME/VAME_CombinedNPW/VAME_CombinedNPW3-Nov24-2020/videos/pose_estimation/')

for file in poseFiles:
    sampleName = file.split('-DC')[0]
    if not os.path.exists(path_to_file + 'data/' + sampleName + '/' + sampleName + '-PE-seq.npy'):
        egocentric_time_series = av.align_demo(path_to_file, sampleName, file_format, crop_size, use_video=False, check_video=False)
        np.save(path_to_file+'data/'+sampleName+'/'+sampleName+'-PE-seq.npy', egocentric_time_series)




