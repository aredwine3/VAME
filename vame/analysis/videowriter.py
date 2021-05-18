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
import cv2 as cv
import tqdm
from vame.util.auxiliary import read_config
import glob

def get_cluster_vid(cfg, path_to_file, file, n_cluster, cluster_method='kmeans', rename=None):
    print("Videos get created for "+file+" ...")
    if cluster_method == 'kmeans':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_km_label_'+'*.npy')[0])
    elif cluster_method == 'GMM':
        labels = np.load(glob.glob(path_to_file+'/'+str(n_cluster)+'_gmm_label_'+'*.npy')[0])
   # labels = np.load(path_to_file+'/'+str(n_cluster)+'_km_label_'+file+'.npy')
    if rename:
        if file.split('_')[-1].startswith('Dec'):
            vidpath = glob.glob(cfg['project_path']+'videos/'+file.split('Dec')[0]+'2020-12*.mp4')[0]
            capture = cv.VideoCapture(vidpath)
        elif file.split('_')[-1].startswith('Jan'):
            vidpath = glob.glob(path_to_file+'/'+str(n_cluster)+file.split('Jan')[0]+'2021-01*.mp4')[0]
            capture = cv.VideoCapture(vidpath)
    elif not rename:
        capture = cv.VideoCapture(cfg['project_path']+'videos/'+file+'.mp4')

    if capture.isOpened():
        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
#        print('width, height:', width, height)

        fps = 25#capture.get(cv.CAP_PROP_FPS)
#        print('fps:', fps)

    cluster_start = cfg['time_window'] / 2
    for cluster in range(n_cluster):
        print('Cluster: %d' %(cluster))
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
        
        if flag == "motif":
            output = os.path.join(path_to_file,"cluster_videos",file+'-motif_%d.avi' %cluster)
        if flag == "community":
            output = os.path.join(path_to_file,"community_videos",file+'-community_%d.avi' %cluster)
            
        video = cv.VideoWriter(output, cv.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))

        if len(cluster_lbl) < cfg['length_of_motif_video']:
            vid_length = len(cluster_lbl)
        else:
            vid_length = cfg['length_of_motif_video']

        for num in tqdm.tqdm(range(vid_length)):
            idx = cluster_lbl[num]
            capture.set(1,idx+cluster_start)
            ret, frame = capture.read()
            video.write(frame)

        video.release()
    capture.release()


def motif_videos(config, model_name, cluster_method="kmeans", n_cluster=[30], rename=None):
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
        
    for cluster in n_cluster:
        print("Cluster size %d " %cluster)
        if rename:
            for file in files:
                suffix=file.split('_')[-1]
                file = file.replace(suffix, rename[suffix])

                path_to_file=cfg['project_path']+'results/'+file+'/'+model_name+'/'+cluster_method+'-'+str(cluster)
                
                if not os.path.exists(path_to_file+'/cluster_videos/'):
                        os.mkdir(path_to_file+'/cluster_videos/')

                get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, cluster_method=cluster_method, rename=rename)

        elif not rename:
            for file in files:
                path_to_file=cfg['project_path']+'results/'+file+'/'+model_name+'/'+cluster_method+'-'+str(cluster)
                
                if not os.path.exists(path_to_file+'/cluster_videos/'):
                        os.mkdir(path_to_file+'/cluster_videos/')

            get_cluster_vid(cfg, path_to_file, file, n_cluster, videoType, flag, cluster_method=cluster_method)
    
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
