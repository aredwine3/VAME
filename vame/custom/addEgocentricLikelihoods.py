#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:59:33 2020

@author: smith
"""

import pandas as pd
import os

orig = pd.read_csv('/d1/studies/VAME/VAME_CombinedNPW/VAME_CombinedNPW_7points_LR1e-3_StepSize30_Gamma0.5-Dec8-2020/videos/pose_estimation/C15_RT_Saline-DC.csv', header=[0,1,2], index_col=0)

cols = orig.columns
newcols = orig.drop('likelihood', axis=1, level=2).columns


directory = '/d1/studies/BSOID2_Ego/train/nolh/'
origDirectory = '/d1/studies/VAME/VAME_CombinedNPW/VAME_CombinedNPW_7points_LR1e-3_StepSize30_Gamma0.5-Dec8-2020/videos/pose_estimation'
files = os.listdir(directory)
for file in files:
    if file.endswith('egocentric.csv'):
        f, e = os.path.splitext(file)
        fo = f.replace('_egocentric', '-DC')
        ldf = pd.read_csv(os.path.join(origDirectory, fo + '.csv'), header=[0,1,2], index_col=0)
        fullpath = os.path.join(directory, file)
        df = pd.read_csv(fullpath, index_col=0)
        df.columns = newcols
        bodyParts = ['nose', 'tail-base', 'spine1', 'spine2', 'spine3', 'forepaw-r', 'forepaw-l', 'hindpaw-r', 'hindpaw-l']
        cat = pd.DataFrame()
    ##    for i in range(len(df.columns)):
   #         bodyParts.append(bp)
 #       bodyParts=list(set(bodyParts))
        for bp in bodyParts:
      #      df[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'likelihood')] = ldf[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'likelihood')]
            cat[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'x')] = df[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'x')]
            cat[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'y')] = df[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'y')]
            cat[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'likelihood')] = ldf[('DLC_resnet50_NPWSep26shuffle1_800000', bp, 'likelihood')]
        cat.columns=ldf.columns
        cat.to_csv(os.path.join(directory, f + '_lh.csv'))

files = os.listdir(directory)
for file in files:
    if file.endswith('lh.csv'):
        fullpath = os.path.join(directory, file)
        df = pd.read_csv(fullpath, header=[0,1,2], index_col=0)
        bodyParts = []
        for i in range(len(df.columns)):
            bp = df.columns[i][1]
            bodyParts.append(bp)
        bodyParts = list(set(bodyParts))
        if 'spine1' not in bodyParts:
            raise NameError("Spine1 not found")
        if df.columns[4][1] != 'spine1':
            print(f + ' spine1 not detected in correct position')


