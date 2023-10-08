#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

from email.quoprimime import header_check
from math import e
import os

import matplotlib

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server

#os.chdir('/d1/software/VAME')
from pathlib import Path
import numpy as np
import pandas as pd
import cv2 as cv
import tqdm
from vame.util.auxiliary import read_config
import glob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
from tqdm import trange
from vame.util import auxiliary as aux
import re

#%%
def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def count_headers(file_path):
    num_headers = 0
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Assuming headers contain only non-numeric values
            if all(not x.replace('.', '', 1).isdigit() for x in row if x):
                num_headers += 1
            else:
                break
    return num_headers
#%%
def get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, fps=30, bins=6, cluster_method='kmeans', extractData=False, symlinks=False):
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
    import cv2 as cv
    import os
    import glob
    import numpy as np


    projectPath=cfg['project_path']
    print("Videos being created for "+file+" ...")
    if cluster_method == 'kmeans':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_km_label_'+'*.npy')[0])
    elif cluster_method == 'GMM':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_gmm_label_'+'*.npy')[0])
    elif cluster_method == 'hmm':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_km_label_'+'*.npy')[0])


    if not symlinks:
        capture = cv.VideoCapture(os.path.join(cfg['project_path'],"videos",file+videoType))
    
    if symlinks:
        # Your original file name
        original_file_name = file + videoType
        
        # Use regular expression to trim off "_RatX" from the file name but keep the extension
        trimmed_file_name = re.sub(r'_Rat\d+', '', original_file_name)
        
        # Construct the full path with the trimmed file name
        trimmed_full_video_path = os.path.join(cfg['project_path'], "real_videos", trimmed_file_name)

        # Resolve symlink to actual file path
        expanded_path = os.path.expandvars(trimmed_full_video_path)
        resolved_path = os.path.realpath(expanded_path)

        print("Full path to video with trimmed name:", trimmed_full_video_path)
        
        capture = cv.VideoCapture(trimmed_full_video_path)

    
    if capture.isOpened():
        print("Video capture successful")  # Debug print 3
        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    else:
        print("Video capture failed")  # Debug print 3
        raise OSError("Could not open OpenCV capture device.")

    
    if extractData:
        if not os.path.exists(os.path.join(path_to_file, 'dlcPoseData')):
            os.mkdir(os.path.join(path_to_file, 'dlcPoseData'))
        
        dataFile = glob.glob(os.path.join(projectPath, 'videos', 'pose_estimation',file+'*.csv'))
        dataFile = dataFile[0] if dataFile else None
            
        if dataFile:
            header_check = count_headers(os.path.join(projectPath, 'videos', 'pose_estimation',file+'*.csv'))
            
            if header_check == 2:
                data = pd.read_csv(dataFile, index_col=0, header=[0,1])
                
            elif header_check == 3:
                data = pd.read_csv(dataFile, index_col=0, header=[0,1,2])
        
    for cluster in range(n_cluster):
        print('Cluster: %d' %(cluster))
        vid_length = cfg['length_of_motif_video']
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
        if cluster_lbl.shape[0] < vid_length:
            vid_length=cluster_lbl.shape[0]-1
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
        if extractData:
            clusterData = data.iloc[used_seqs,:]
            clusterData.to_csv(os.path.join(path_to_file,'dlcPoseData', file+'_DLC_Results_Cluster'+str(cluster)+'.csv'))
               
        if len(used_seqs) > vid_length:
            used_seqs = used_seqs[:vid_length]
            
        if flag == "motif":
            output = os.path.join(path_to_file,"cluster_videos",file+'-motif_%d_longestSequences_binned%d.avi' %(cluster,bins))
        elif flag == "community":
            output = os.path.join(path_to_file,"community_videos",file+'-community_%d_longestSequences_binned%d.avi' %(cluster,bins))
            
        if os.path.exists(os.path.join(path_to_file,"cluster_videos",file+'-motif_%d_longestSequences_binned%d.avi' %(cluster,bins))):
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

#%%
def motif_videos(config, model_name, videoType='.mp4', fps=30, bins=6, cluster_method="kmeans", extractData=False, symlinks=False):
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
        Method used for clustering. Options are 'kmeans' or 'hmm'
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
    symlinks = symlinks
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

        get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, fps=fps, bins=bins, cluster_method=cluster_method, extractData=extractData, symlinks=symlinks)

    print("All videos have been created!")
    
#%%
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
    
    
#%%
def reformat_aligned_array(array, dlc_data, output_dir, filename):
  #  arr = np.load(array)
    dfa = pd.DataFrame(array).T
    num_insertions=(dfa.shape[1])//2
 #   n, e = os.path.splitext(os.path.basename(dlc_data))
    for i in range(num_insertions):
        insert_position = (i + 1) * 2 + i  # Calculate the position for insertion
        new_column_name = 'likelihood_'+str(i)
        dfa.insert(insert_position, new_column_name, np.ones(dfa.shape[0]))
    dfa.columns=dlc_data.columns
    if output_dir:
        dfa.to_csv(os.path.join(output_dir, filename+'_Aligned.csv'))
    return dfa

#%% This is taken exactly from DeepLabCut, credit to the authors www.github.com/deeplabcut/deeplabcut
def create_video_with_keypoints_only(
    df,
    output_name,
    ind_links=None,
    pcutoff=0.6,
    dotsize=8,
    alpha=0.7,
    background_color="k",
    skeleton_color="navy",
    color_by="bodypart",
    colormap="viridis",
    fps=25,
    dpi=200,
    codec="h264",
):
    bodyparts = df.columns.get_level_values("bodyparts")[::3]
    bodypart_names = bodyparts.unique()
    n_bodyparts = len(bodypart_names)
    nx = int(np.nanmax(df.xs("x", axis=1, level="coords")))
    ny = int(np.nanmax(df.xs("y", axis=1, level="coords")))

    n_frames = df.shape[0]
    xyp = df.values.reshape((n_frames, -1, 3))

    if color_by == "bodypart":
        map_ = bodyparts.map(dict(zip(bodypart_names, range(n_bodyparts))))
        cmap = plt.get_cmap(colormap, n_bodyparts)
    elif color_by == "individual":
        try:
            individuals = df.columns.get_level_values("individuals")[::3]
            individual_names = individuals.unique().to_list()
            n_individuals = len(individual_names)
            map_ = individuals.map(dict(zip(individual_names, range(n_individuals))))
            cmap = plt.get_cmap(colormap, n_individuals)
        except KeyError as e:
            raise Exception(
                "Coloring by individuals is only valid for multi-animal data"
            ) from e
    else:
        raise ValueError(f"Invalid color_by={color_by}")

    prev_backend = plt.get_backend()
    plt.switch_backend("agg")
    fig = plt.figure(frameon=False, figsize=(nx / dpi, ny / dpi))
    ax = fig.add_subplot(111)
    scat = ax.scatter([], [], s=dotsize ** 2, alpha=alpha)
    coords = xyp[0, :, :2]
    coords[xyp[0, :, 2] < pcutoff] = np.nan
    scat.set_offsets(coords)
    colors = cmap(map_)
    scat.set_color(colors)
    segs = coords[tuple(zip(*tuple(ind_links))), :].swapaxes(0, 1) if ind_links else []
    coll = LineCollection(segs, colors=skeleton_color, alpha=alpha)
    ax.add_collection(coll)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=background_color, transform=ax.transAxes, zorder=-1
        )
    )
    ax.invert_yaxis()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    writer = FFMpegWriter(fps=fps, codec=codec)
    with writer.saving(fig, output_name, dpi=dpi):
        writer.grab_frame()
        for index, _ in enumerate(trange(n_frames - 1), start=1):
            coords = xyp[index, :, :2]
            coords[xyp[index, :, 2] < pcutoff] = np.nan
            scat.set_offsets(coords)
            if ind_links:
                segs = coords[tuple(zip(*tuple(ind_links))), :].swapaxes(0, 1)
            coll.set_segments(segs)
            writer.grab_frame()
    plt.close(fig)
    plt.switch_backend(prev_backend)

#%%
def create_egocentric_videos(config, output_dir, colormap='viridis', dotsize=4, fps=25, num_vids=None, vid_index=None):
    cfg=aux.read_config(config)
    if not output_dir:
        output_dir=cfg['project_path']
    vids = cfg['video_sets']
    if not num_vids:
        num_vids = len(vids)
        print("Creating egocentrically aligned video for " + str(num_vids) + " videos")
    if vid_index:
        vids = [vids[vid_index]]
    for v in vids[:num_vids]:
        arr = np.load(os.path.join(cfg['project_path'], 'data', v, v+'-PE-seq.npy'))
        dlcs = glob.glob(os.path.join(cfg['project_path'], 'videos', 'pose_estimation', v+'*'))
        n, e = os.path.splitext(os.path.basename(dlcs[0]))
        if e=='.csv':
            dlc_data = pd.read_csv(dlcs[0], index_col=0, header=[0,1,2])
        elif e=='.h5':
            dlc_data = pd.read_hdf(dlcs[0])
        dfa = reformat_aligned_array(arr, dlc_data, output_dir, n)
        print("Creating video for "+v)
        create_video_with_keypoints_only(dfa, os.path.join(output_dir, v+'_aligned.mp4'), dotsize=dotsize, colormap=colormap, fps=fps)
        
