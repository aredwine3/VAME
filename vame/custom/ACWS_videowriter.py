#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2 as cv
import tqdm
from vame.util.auxiliary import read_config
import glob

def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, fps=30, bins=6, cluster_method='kmeans'):
    """This creates motif videos based on the longest sequences of each motif, rather than the first frames in that motif (the default in VAME).
    You can limit the length of each sequence used with 'bins', this parameter sets the minimum number of distinct examples that will be sampled
    in the video (if that many examples exist).
    
    Parameters
    ----------
    cfg : dict
        VAME config dictionary.
    path_to_file : str
        Path to video file to process (base directory without filename).
    file : str
        Name of file to process within path_to_file (without extension).
    n_cluster : int
        Number of clusters (from config file).
    flag : str
        'motif' or 'community', depending on type of video being created. Only tested with 'motif' so far.
    videoType: str
        Extension of video file.
    fps : int (optional, default 30)
        Framerate for output video.
    bins : int (optional, default 6)
        Number of distinct example to sample. For example if length_of_motif_video in config.yaml is set to 1000, and bins=6, a maximum ~166 (1000/6)
        frames of each sequence will be used in creating the motif video. Prevents motif videos from being one long behavioral sequence.
    cluster_method : str (optional, default 'kmeans')
        'kmeans' or 'GMM'
    
    Returns
    -------
    None. Saves motif videos.
    
    """
    
    print("Videos being created for "+file+" ...")
    if cluster_method == 'kmeans':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_km_label_'+'*.npy')[0])
    elif cluster_method == 'GMM':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_gmm_label_'+'*.npy')[0])
    capture = cv.VideoCapture(os.path.join(cfg['project_path'],"videos",file+videoType))  

    if capture.isOpened():
        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    else:
        raise OSError("Could not open OpenCV capture device.")

    for cluster in range(n_cluster):
        print('Cluster: %d' %(cluster))
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
        cons_lbl = consecutive(cluster_lbl, 1)
        cons_df = pd.DataFrame(cons_lbl)
        try:
            startFrames = cons_df[0].tolist()
        except KeyError:
            startFrames=0
        cons_df = cons_df.T
        cons_counts = pd.DataFrame(cons_df.count())
        cons_counts.columns=['length']
        cons_counts['startFrame']=startFrames
        cons_counts.sort_values('length', inplace=True, ascending=False)
        cons_counts.reset_index(inplace=True)
        frames = cons_counts['startFrame'].tolist()
        lengths = cons_counts['length'].tolist()

        vid_length = cfg['length_of_motif_video']
        used_seqs=[]
        while len(used_seqs)<vid_length:
            if lengths[0]==0:
                break
            for i in frames:
                if len(used_seqs)>=vid_length:
                    break
                idx = frames.index(i)
                length = lengths[idx]
                endFrame = i+length
                seq = list(range(i, endFrame))
                if len(seq)<vid_length//bins:
                    used_seqs.extend(seq)
                else:
                    seq = seq[:vid_length//bins]
                    used_seqs.extend(sorted(seq))
               
        if len(used_seqs) > vid_length:
            used_seqs = used_seqs[:vid_length]
            
        if flag == "motif":
            output = os.path.join(path_to_file,"cluster_videos",file+'-motif_%d_longestSequences_binned'+str(bins)+'.avi' %cluster)
        elif flag == "community":
            output = os.path.join(path_to_file,"community_videos",file+'-community_%d_longestSequences_binned'+str(bins)+'.avi' %cluster)
            
        if os.path.exists(os.path.join(path_to_file,"cluster_videos",file+'-motif_%d_longestSequences_binned'+str(bins)+'.avi' %cluster)):
            print("Video for cluster %d already found, skipping..." %cluster)
            continue
        else:
            video = cv.VideoWriter(output, cv.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))
    
            if len(used_seqs) < cfg['length_of_motif_video']:
                vid_length = len(used_seqs)
            else:
                vid_length = cfg['length_of_motif_video']
    
            for num in tqdm.tqdm(range(vid_length)):
                idx = used_seqs[num]
                capture.set(1,idx)
                ret, frame = capture.read()
                video.write(frame)
    
            video.release()
    capture.release()


def motif_videos(config, model_name, videoType='.mp4', fps=25, bins=6, cluster_method="kmeans"):
    """Create custom motif videos. This differs from the function in the main vame repository in that 
    rather than the first frames 1000 frames (or whatever number assigned in config.yaml) of each cluster,
    motif videos will sample the longest sequential number of frames in the same behavioral cluster.
    The 'bins' parameter allows manual diversifying of the motif videos, as it will represent
    the minimum number of examples sampled from. For example if creating a 1000 frame motif video with
    6 bins, each bin will be ~167 frames long, to prevent motif videos from having only one long continuous
    behavior video. Milage may very, but this helps with my data, and feedback is welcome.
    
    Parameters
    ----------
    config : string
        Path to config.yaml file for project.
    model_name : string
        Name of model (subdirectory of 'results/')
    videoType : string, optional (default '.mp4')
        Video format analyzed.
    fps : int (optional, default 30)
        Frames per second for output result video.
    bins : int (optional, default 6)
        Minimum number of bins different behavioral epochs to include in motif videos.
    cluster_method : string (optional, default 'kmeans')
        Method used for clustering.
    rename : tuple (optional, default None)
        In very early trial testing, but allows analysis of videos that have been renamed, 
        or that you want to rename during analysis.
        
    Returns
    -------
    Creates files with examples of each segmented behavior,
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    flag = 'motif'
    
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
        
    for file in files:
        path_to_file=os.path.join(cfg['project_path'], 'results/',file,model_name,cluster_method+'-'+str(n_cluster),'')
        
        if not os.path.exists(path_to_file+'/cluster_videos/'):
            os.mkdir(path_to_file+'/cluster_videos/')

        get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, fps=fps, bins=bins, cluster_method=cluster_method)

    print("All videos have been created!")
    
    
def community_videos(config, videoType='.mp4'):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    flag = 'community'
    
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

    print("Cluster size is: %d " %n_cluster)
    for file in files:
        path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")
        if not os.path.exists(os.path.join(path_to_file,"community_videos")):
            os.mkdir(os.path.join(path_to_file,"community_videos"))

        get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag)
    
    print("All videos have been created!")
