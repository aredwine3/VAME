#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:52:04 2020

@author: luxemk
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from vame.util.auxiliary import read_config
from vame.analysis.behavior_structure import get_adjacency_matrix, get_transition_matrix
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import shutil

def trimFrames(directory, begin=1500, end=1500):
    """Crop csv or data files within specific frames. Good for removing unuseful frames from beginning, end, or both.
    
    Parameters
    ----------
    directory : string
        Path to directory containing data files to process.
    startFrame : int (optional, default None)
        Starting frame for trimmed data.
    stopFrame : int (optional, default None)
        Ending frame for trimmed data.
    """
    files = os.listdir(directory)
    saveDir = os.path.join(directory, 'trimmedData/')
    startFrame=begin
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    for f in files:
        if f.endswith('.csv'):
            n, e = os.path.splitext(f)
            fullpath = os.path.join(directory, f)
            df = pd.read_csv(fullpath, header = [0,1,2], index_col=0)
            stopFrame=int(df.shape[0]-int(end))
            df = df[startFrame:stopFrame]
            df.to_csv(os.path.join(saveDir, n + '_trimmed.csv'))
        elif f.endswith('.h5'):
            n, e = os.path.splitext(f)
            fullpath = os.path.join(directory, f)
            df = pd.read_hdf(fullpath)
            df = df[startFrame:stopFrame]
            df.to_hdf(os.path.join(saveDir, n + '_trimmed.csv'), key='df_with_missing')

def listBodyParts(config):
    cfg = read_config(config)
    projectPath = cfg['project_path']
    dataPath = os.path.join(projectPath, 'videos/pose_estimation/')
    dataFiles = os.listdir(dataPath)
    df = pd.read_csv(os.path.join(dataPath, dataFiles[0]), index_col=0, header=[0,1,2])
    cols = df.columns.tolist()
    bodyParts = []
    for col in cols:
        bp = col[1]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts))
    return bodyParts

def makeEgocentricCSV(h5Path, bodyPart):
    """Docstring:
        Deprecated. Use alignVideos.alignVideo() instead.
    """
    directory = '/'.join(h5Path.split('/')[:-1])
    fileName = h5Path.split('/')[-1]#.split('DLC')[0]
    f, e = os.path.splitext(fileName)
    df = pd.read_hdf(h5Path)
    cols = df.columns
    scorerName=cols[0][0]
    bodyParts = []
    for col in cols:
        bp = col[1]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts)) #remove duplicates
    df_ego = df
    bodyParts_norm = bodyParts
    bodyParts_norm.remove(bodyPart)
    for bp in bodyParts_norm: #normalize bodyparts by subtracting one from all 
        df_ego[(scorerName, bp, 'x')] = df_ego[(scorerName, bp, 'x')] - df_ego[(scorerName, bodyPart, 'x')] #for x position
        df_ego[(scorerName, bp, 'y')] = df_ego[(scorerName, bp, 'y')] - df_ego[(scorerName, bodyPart, 'y')] #for y position

    if not os.path.exists(os.path.join(directory, 'egocentric/')):
        os.mkdir(os.path.join(directory, 'egocentric/'))
    df_ego.to_csv(os.path.join(directory, 'egocentric/' + f + '_egocentric.csv'))

def makeEgocentricCSV_Center(h5Path, bodyPart1, bodyPart2, drop=None):
    """
    Docstring:
        Deprecated. Use alignVideos.alignVideo() instead.
    """
    directory = '/'.join(h5Path.split('/')[:-1])
    fileName = h5Path.split('/')[-1]#.split('DLC')[0]
    f, e = os.path.splitext(fileName)
    df = pd.read_hdf(h5Path)
    cols = df.columns
    newCols = cols.droplevel(level=0) #drops the 'scorer' level from the DLC data MultiIndex
    df.columns = newCols
    if drop: #if you want to drop a bodypart from the data you can use this argument
        df.drop(labels=drop, axis=1, level=0, inplace=True)
    bodyParts = []
    cols = df.columns
    for col in cols:
        bp = col[0]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts)) #remove duplicates
    df_ego = df
    a_x = pd.DataFrame([df_ego[(bodyPart1, 'x')], df_ego[(bodyPart2, 'x')]]).T
    a_y = pd.DataFrame([df_ego[(bodyPart1, 'y')], df_ego[(bodyPart2, 'y')]]).T
    df_ego[('egoCentricPoint', 'x')] = np.mean(a_x, axis=1)
    df_ego[('egoCentricPoint', 'y')] = np.mean(a_y, axis=1)
    for bp in bodyParts: #normalize to mean x/y of two specified bodyparts
        df_ego[(bp, 'x')] = df_ego[(bp, 'x')] - df_ego[('egoCentricPoint', 'x')]
        df_ego[(bp, 'y')] = df_ego[(bp, 'y')] - df_ego[('egoCentricPoint', 'y')]
    if not os.path.exists(os.path.join(directory, 'egocentric/')):
        os.mkdir(os.path.join(directory, 'egocentric/'))
    df_ego.to_csv(os.path.join(directory, 'egocentric/' + f + '_egocentric_centered.csv'))

    
def csv_to_numpy(projectPath, csvPath, pcutoff=.99):
    """
    Convert CSV to NumPy array.
    
    Parameters
    ----------
    projectPath : string
        path to project folder.
    csvPath : string
        path to CSV
    pcutoff : float (optional, default .99)
        p-cutoff for likelihood, coordinates below this will be set to NaN.
    """

    # Read in your .csv file, skip the first two rows and create a numpy array
    data = pd.read_csv(csvPath, skiprows = 1)
    fileName = csvPath.split('/')[-1].split('DLC')[0]
    f, e = os.path.splitext(fileName)
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
        seq[con_arr[idx,:]<pcutoff] = np.NaN
        body_position_nan.append(seq)
    
    final_positions = np.array(body_position_nan)
    # save the final_positions array with np.save()
    np.save(os.path.join(projectPath, 'data/' + f + '/' + f + "-PE-seq.npy"), final_positions)

def combineBehavior(config, save=True, cluster_method='kmeans', legacy=False):
    """
    Docstring:
        Combines motif usage for all samples into one CSV file.
        
    Parameters
    ----------
    config : string 
        path to config file
    save : bool
    n_cluster : int 
        number of clusters to analyze (behavioral segmentation data must exist)
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    project_path = cfg['project_path']
    n_cluster=cfg['n_cluster']
    model_name = cfg['model_name']
    files = []
    if cfg['all_data'] == 'No':
        all_flag = input("Do you want to write motif videos for your entire dataset? \n"
                     "If you only want to use a specific dataset type filename: \n"
                     "yes/no/filename ")
    else: 
        all_flag = 'yes'
        
    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)
            
    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        files.append(all_flag)
    
    cat = pd.DataFrame()
    for file in files:
        if legacy:
            arr = np.load(os.path.join(project_path, 'results/' + file + '/VAME_NPW/kmeans-' + str(n_cluster) + '/behavior_quantification/motif_usage.npy'))
        elif not legacy:
            arr = np.load(os.path.join(project_path, 'results',file,model_name,cluster_method+'-'+str(n_cluster),'motif_usage_'+file+'.npy'))
        df = pd.DataFrame(arr, columns=[file])
        cat = pd.concat([cat, df], axis=1)
    if save:
        cat.to_csv(os.path.join(project_path, 'CombinedMotifUsage.csv'))
    return cat

def parseIVSA(config, groups, presession=True):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    projectPath = cfg['project_path']
    files = cfg['video_sets']
    cat = combineBehavior(config)
    for group in groups:
        groupData=pd.DataFrame()
        if presession:
            preData = pd.DataFrame()
        for file in files:
            if group in file:
                if not 'Presession' in file:
                    groupData[file]=cat[file]
                elif 'Presession' in file:
                    preData[file]=cat[file]
            groupData.to_csv(os.path.join(projectPath, group + '_MotifUsage.csv'))
            if presession:
                preData.to_csv(os.path.join(projectPath, 'Presession_'+group+'MotifUsage.csv'))
                
def extractResults(projectPath, expDate, group1, group2, modelName, n_clusters, cluster_method='kmeans', phases=None):
    """Docstring:
    Compares motif usage between two groups with t-test.
    
    
    Parameters
    ----------
    projectPath : string
        Path to project folder
    group1 : list
        List of subject ID strings.
    group2 : list
        List of subject ID strings.
    modelName : string
        Name of model to use.
    n_clusters : int
        Number of segmented clusters to analyze.
    phases (optional): 
        Experimental phases that are suffixes for data files.

    Returns
    -------
    Pandas DataFrame with results. Also saves CSV.

    """
    if not os.path.exists(os.path.join(projectPath, str(n_clusters) + 'Clusters/')):
        os.mkdir(os.path.join(projectPath, str(n_clusters) + 'Clusters/'))
    saveDir = os.path.join(projectPath, str(n_clusters) + 'Clusters/')
    samples = os.listdir(os.path.join(projectPath, 'results/'))
    cat = pd.DataFrame()
    for sample in samples:
        clu_arr = np.load(os.path.join(projectPath, 'results/' + sample + '/' + modelName + '/' + cluster_method + '-' + str(n_clusters) + '/behavior_quantification/motif_usage.npy'))
        clu = pd.DataFrame(clu_arr)
        clu.columns=[sample]
        cat = pd.concat([cat, clu], axis=1)
 #   cat.to_csv(os.path.join(projectPath, modelName + '_Results.csv'))

    df1=pd.DataFrame()
    df2=pd.DataFrame()


    for col in cat.columns:
        if col[:6] in group1:
            df1[col]=cat[col]
        elif col[:5] in group1:
            df1[col]=cat[col]
        elif col[:6] in group2:
            df2[col]=cat[col]
        elif col[:5] in group2:
            df2[col]=cat[col]
        else:
            print(str(col) + " not found in either")

    df1['Group1_mean'] = np.mean(df1, axis=1)
    df1['Group1_sem']=(np.std(df1, axis=1)/np.sqrt((df1.shape[1]-1)))
    df2['Group2_mean'] = np.mean(df2, axis=1)
    df2['Group2_sem']=(np.std(df2, axis=1)/np.sqrt((df2.shape[1]-1)))
    
    df1.to_csv(os.path.join(saveDir, 'Group1_' + modelName + '_' + str(n_clusters) + 'Clusters.csv'))
    df2.to_csv(os.path.join(saveDir, 'Group2_' + modelName + '_' + str(n_clusters) + 'Clusters.csv'))
    comb = pd.concat([df1, df2], axis=1)
    comb.to_csv(os.path.join(saveDir, 'Combined_' + modelName + '_' + str(n_clusters) + 'Clusters_Results.csv'))

#    cols = list(df1.columns)
#    for col in cols:
#        if col.endswith(expDate):
#            newCol = '_'.join(col.split('_')[:-1])
#            i = cols.index(col)
#            cols.remove(col)
#            cols.insert(i, newCol)
#    df1.columns=cols
    
#    cols = list(df2.columns)
#    for col in cols:
#        if col.endswith(expDate):
#            newCol = '_'.join(col.split('_')[:-1])
#            i = cols.index(col)
#            cols.remove(col)
#            cols.insert(i, newCol)
#    df2.columns=cols
    if phases:
        for phase in phases:
            df1_split = pd.DataFrame()
            for col in df1.columns:
                if col.endswith(phase):
                    df1_split[col]=df1[col]
            df1_split['Group1_mean'] = np.mean(df1_split, axis=1)
            df1_split['Group1_sem']=(np.std(df1_split, axis=1)/np.sqrt((df1_split.shape[1]-1)))
            df1_split.to_csv(os.path.join(saveDir, 'Group1_' + phase + '_' + modelName + '_' + str(n_clusters) + 'Clusters_Results.csv'))
                    
            df2_split = pd.DataFrame()
            for col in df2.columns:
                if col.endswith(phase):
                    df2_split[col]=df2[col]
            df2_split['Group2_mean'] = np.mean(df2_split, axis=1)
            df2_split['Group2_sem']=(np.std(df2_split, axis=1)/np.sqrt((df2_split.shape[1]-1)))
            df2_split.to_csv(os.path.join(saveDir, 'Group2_' + phase + '_' + modelName + '_' + str(n_clusters) + 'Clusters_Results.csv'))
               
            results = pd.concat([df1_split, df2_split], axis=1)
            
            df1_arr = np.array(df1_split.iloc[:,:-2])
            df2_arr = np.array(df2_split.iloc[:,:-2])
            if len(df1_arr) > 0 and len(df2_arr) > 0:
                pvals = np.array([ttest_ind(df1_arr[i,:], df2_arr[i,:], nan_policy='omit')[1] for i in range(df1_arr.shape[0])])
                fdr_pass,qvals,_,_ = multipletests(pvals, method='fdr_bh',alpha=0.05)
        
                results['p-value'] = pvals
                results['q-value'] = qvals
                results.to_csv(os.path.join(saveDir, 'Combined_' + phase + '_' + modelName + '_' + str(n_clusters) + 'Clusters_Results.csv'))
                
                summary = pd.DataFrame()
                summary['Group1_Mean'] = df1_split['Group1_mean']
                summary['Group1_SEM'] = df1_split['Group1_sem']
                summary['Group2_Mean'] = df2_split['Group2_mean']
                summary['Group2_SEM'] = df2_split['Group2_sem']
                summary['p-value'] = results['p-value']
                summary['q-value'] = results['q-value']
                summary.to_csv(os.path.join(saveDir, phase + '_SummaryStatistics_' + str(n_clusters) + 'clusters.csv'))
            else:
                print("No results found for both groups found in phase " + str(phase))
    elif not phases:
        results = pd.concat([df1, df2], axis=1)
        
        df1_arr = np.array(df1.iloc[:,:-2])
        df2_arr = np.array(df2.iloc[:,:-2])
        pvals = np.array([ttest_ind(df1_arr[i,:], df2_arr[i,:], nan_policy='omit')[1] for i in range(df1_arr.shape[0])])
        fdr_pass,qvals,_,_ = multipletests(pvals, method='fdr_bh',alpha=0.05)
        
        results['p-value'] = pvals
        results['q-value'] = qvals
        results.to_csv(os.path.join(saveDir, 'Combined_' + modelName + '_' + str(n_clusters) + 'Clusters_Results.csv'))
        
        summary = pd.DataFrame()
        summary['Group1_Mean'] = df1['Group1_mean']
        summary['Group1_SEM'] = df1['Group1_sem']
        summary['Group2_Mean'] = df2['Group2_mean']
        summary['Group2_SEM'] = df2['Group2_sem']
        summary['p-value'] = results['p-value']
        summary['q-value'] = results['q-value']
        summary.to_csv(os.path.join(saveDir, 'SummaryStatistics_' + str(n_clusters) + 'clusters.csv'))
     
        
def selectLimbs(projectPath, suffix):
    """Docstring:
        
    Parameters
    ----------
    projectPath : string
        Path to project folder.
    suffix : string
        Suffix to add to file name when saving.
    """
    poseFiles = os.listdir(os.path.join(projectPath, 'videos/pose_estimation/'))
    for file in poseFiles:
        fullpath = os.path.join(projectPath, 'videos/pose_estimation/' + file)
        f, e = os.path.splitext(file)
        df = pd.read_csv(fullpath, header=[0,1,2], index_col=0)
        cols = df.columns
        newCols = cols.droplevel(level=0) #drops the 'scorer' level from the h5 MultiIndex
        df.columns = newCols
        bodyParts_f = ['forepaw-l', 'forepaw-r']
        bodyParts_h = ['hindpaw-l', 'hindpaw-r']
        bps_f = []
        aves_f = []
        for bp in bodyParts_f:
            ave = np.mean(df[(bp, 'likelihood')])
            aves_f.append(ave)
            bps_f.append(bp)
        bps_h = []
        aves_h = []
        for bp in bodyParts_h:
            ave=np.mean(df[(bp, 'likelihood')])
            aves_h.append(ave)
            bps_h.append(bp)
        lz_f = list(zip(aves_f, bps_f))
        lz_h = list(zip(aves_h, bps_h))
        use = ['nose', 'tail-base', 'spine1', 'spine2', 'spine3']
        lists = [lz_f, lz_h]
        for lz in lists:
            if lz[0][0] > lz[1][0]:
                use.append(lz[0][1])
            else:
                use.append(lz[1][1])
        df2 = pd.DataFrame()
        cat = pd.DataFrame()
        for bp in use:
            l = [bp, bp, bp]
            i = ['x', 'y', 'likelihood']
            lz = list(zip(l,i))
            ind = pd.MultiIndex.from_tuples(lz)
        #    df2.columns=[ind]
            df2 = pd.DataFrame(df[bp].values, columns=[lz])
            cat = pd.concat([cat, df2], axis=1)
        usedBPs = list(set([cat.columns[x][0][0] for x in range(len(cat.columns))]))
        bodyParts = []
        for col in newCols:
            bp = col[0]
            bodyParts.append(bp)
        bps = list(set(bodyParts))
        dropBPs=[]
        for bp in bps:
            if bp not in usedBPs:
                dropBPs.append(bp)
        for bp in dropBPs:
            df.drop(bp, axis=1, inplace=True)
            
        old_cols = df.columns.to_frame()
        old_cols.insert(0, 0, 'DLC_resnet50_NPWSep26shuffle1_800000')
        df.columns = old_cols
        ind = pd.MultiIndex.from_tuples(df.columns)
        df.columns=ind
        df.to_csv(os.path.join(projectPath, 'videos/pose_estimation/' + f + suffix + '.csv'))


def dropBodyParts(config, bodyParts):
    """
    Drops specified body parts from CSV file. Creates a new folder called 'original/' and saves original CSVs there,
    and overwrites the CSV in projectPath.
    
    
    Parameters
    ----------
    projectPath : string
        Path to project folder
    bodyParts : list
        List of bodyparts to drop.
    """
    cfg = read_config(config)
    projectPath=cfg['project_path']
    poseFiles = os.listdir(os.path.join(projectPath, 'videos/pose_estimation/'))
    if not os.path.exists(os.path.join(projectPath, 'videos/pose_estimation/original/')):
        os.mkdir(os.path.join(projectPath, 'videos/pose_estimation/original/'))
    for file in poseFiles:
        if file.endswith('.csv') and not file.startswith('.'):
            fullpath = os.path.join(projectPath, 'videos/pose_estimation/' + file)
            if not os.path.exists(os.path.join(projectPath, 'videos/pose_estimation/original/', file)):
                f, e = os.path.splitext(file)
                df = pd.read_csv(fullpath, header=[0,1,2], index_col=0)
                df.to_csv(os.path.join(projectPath, 'videos/pose_estimation/original/', file))
                dropList = []
                scorer=df.columns[0][0]
                if not isinstance(bodyParts, list):
                    bodyParts = [bodyParts]
                for part in bodyParts:
                    tup1 = (scorer, part, 'x')
                    tup2 = (scorer, part, 'y')
                    tup3 = (scorer, part, 'likelihood')
                    dropList.append(tup1)
                    dropList.append(tup2)
                    dropList.append(tup3)
                df.drop(labels=dropList, axis=1, inplace=True)
                df.to_csv(os.path.join(projectPath, 'videos/pose_estimation/' + file))

def plotAverageTransitionMatrices(config, group1, group2=None, g1name='Group1', g2name='Group2', cluster_method='kmeans'):
    cfg = read_config(config)
    projectPath = cfg['project_path']
    n_cluster=cfg['n_cluster']
    model_name=cfg['model_name']
    resultDir = os.path.join(projectPath, 'results')    
    g1tms = []
    g2tms = []
    files = cfg['video_sets']
    for file in files:
        if file in group1:
            labels = np.load(os.path.join(resultDir, file, model_name, cluster_method+'-'+str(n_cluster), str(n_cluster)+'_km_label_'+file+'.npy'))
            am, tm = get_adjacency_matrix(labels, n_cluster)
            g1tms.append(tm)
        if group2:
         if file in group2:
            labels = np.load(os.path.join(resultDir, file, model_name, cluster_method+'-'+str(n_cluster), str(n_cluster)+'_km_label_'+file+'.npy'))
            am, tm = get_adjacency_matrix(labels, n_cluster)
            g2tms.append(tm)
    g1stack = np.stack(g1tms, axis=2)
    g1ave = np.mean(g1stack, axis=2)
    fig = plt.figure(figsize=(15,10))
    sns.heatmap(g1ave, annot=True)
    plt.xlabel("Next frame behavior", fontsize=16)
    plt.ylabel("Current frame behavior", fontsize=16)
    plt.title("Averaged Transition matrix of {} clusters".format(tm.shape[0]), fontsize=18)
    fig.savefig(os.path.join(projectPath, g1name+'_AverageTransitionMatrix_' + str(n_cluster) + 'clusters_'+str(cluster_method)+'.png'), bbox_inches='tight')
    print("Figure saved to " + os.path.join(projectPath, g1name+'_AverageTransitionMatrix_' + str(n_cluster) + 'clusters_'+str(cluster_method)+'.png'))
    if group2:
        g2stack = np.stack(g2tms, axis=2)
        g2ave = np.mean(g2stack, axis=2)
        fig = plt.figure(figsize=(15,10))
        sns.heatmap(g2ave, annot=True)
        plt.xlabel("Next frame behavior", fontsize=16)
        plt.ylabel("Current frame behavior", fontsize=16)
        plt.title("Averaged Transition matrix of {} clusters".format(tm.shape[0]), fontsize=18)
        fig.savefig(os.path.join(projectPath, g2name+'_AverageTransitionMatrix_' + str(n_cluster) + 'clusters_'+str(cluster_method)+'.png'), bbox_inches='tight')
        print("Figure saved to " + os.path.join(projectPath, g2name+'_AverageTransitionMatrix_' + str(n_cluster) + 'clusters_'+str(cluster_method)+'.png'))


def plotLoss(config, suffix=None):
    cfg = read_config(config)
    projectPath = cfg['project_path']
    lossPath = os.path.join(projectPath, 'model', 'model_losses')
    df = pd.read_csv(os.path.join(lossPath, 'VAME_LossesSummary.csv'), index_col=0)
    fig = df.plot(y=['Train_losses', 'Test_losses', 'MSE_losses']).get_figure()
    fig.savefig(os.path.join(lossPath, 'ModelLosses.png'))
    
    
    
    
    
    
    