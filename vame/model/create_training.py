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
import matplotlib

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server


import numpy as np
import pandas as pd
from pathlib import Path
import scipy.signal
from datetime import date
from scipy.stats import iqr # type: ignore
import matplotlib.pyplot as plt

from vame.util.auxiliary import read_config


#Helper function to return indexes of nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

#Interpolates all nan values of given array
def interpol(arr):
    y = np.transpose(arr)
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    arr = np.transpose(y)
    return arr

def plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med): #, anchor_1, anchor_2):
    plot_X_orig = np.concatenate(X_true, axis=0).T
    plot_X_med = X_med.copy()
    iqr_cutoff = cfg['iqr_factor']*iqr_val
    
    plt.figure()
    plt.plot(plot_X_orig.T)
    plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
    plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
    plt.title("Full Signal z-scored")
    plt.legend()
    if num_frames > 1000:
        rnd = np.random.choice(num_frames)
        
        plt.figure()
        plt.plot(plot_X_med[:,rnd:rnd+1000].T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Filtered signal z-scored")
        plt.legend()
        
        plt.figure()
        plt.plot(plot_X_orig[:,rnd:rnd+1000].T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Original signal z-scored")
        plt.legend()
        
        plt.figure()
        plt.plot(plot_X_orig[:,rnd:rnd+1000].T, 'g', alpha=0.5)
        plt.plot(plot_X_med[:,rnd:rnd+1000].T, '--m', alpha=0.6)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Overlayed z-scored")
        plt.legend()
        
        # plot_X_orig = np.delete(plot_X_orig.T, anchor_1, 1)
        # plot_X_orig = np.delete(plot_X_orig, anchor_2, 1)
        # mse = (np.square(plot_X_orig[rnd:rnd+1000, :] - plot_X_med[:,rnd:rnd+1000].T)).mean(axis=0)
        
        
    else:
        plt.figure()
        plt.plot(plot_X_med.T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Filtered signal z-scored")
        plt.legend()
        
        plt.figure()
        plt.plot(plot_X_orig.T)
        plt.axhline(y=iqr_cutoff, color='r', linestyle='--', label="IQR cutoff")
        plt.axhline(y=-iqr_cutoff, color='r', linestyle='--')
        plt.title("Original signal z-scored")
        plt.legend()
        
    print("Please run the function with check_parameter=False if you are happy with the results")

def traindata_aligned(cfg, files, testfraction, num_features, savgol_filter, check_parameter):
    
    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    
    if check_parameter == True:
        X_true = []
        files = [files[0]]
        
    for file in files:
        try: 
            print("z-scoring of file %s" %file)
            path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
            
            # 1. File Loading
            try:
                data = np.load(path_to_file)
            except Exception as e:
                print(f"Error occurred while loading {file}. Skipping this file.")
                print(e)
                continue
           
            # 2. Z-Scoring
            try:
                X_mean = np.mean(data,axis=None)
                X_std = np.std(data, axis=None)
                X_z = (data.T - X_mean) / X_std
            except Exception as e:
                print(f"Error occurred while z-scoring {file}. Skipping this file.")
                print(e)
            
            # Introducing artificial error spikes
            # rang = [1.5, 2, 2.5, 3, 3.5, 3, 3, 2.5, 2, 1.5]
            # for i in range(num_frames):
            #     if i % 300 == 0:
            #         rnd = np.random.choice(12,2)
            #         for j in range(10):
            #             X_z[i+j, rnd[0]] = X_z[i+j, rnd[0]] * rang[j]
            #             X_z[i+j, rnd[1]] = X_z[i+j, rnd[1]] * rang[j]
                    
            if check_parameter == True:
                X_z_copy = X_z.copy()
                X_true.append(X_z_copy)
                
            if cfg['robust'] == True:
                iqr_val = iqr(X_z)
                print("IQR value: %.2f, IQR cutoff: %.2f" %(iqr_val, cfg['iqr_factor']*iqr_val))
                X_z[(X_z > cfg['iqr_factor']*iqr_val) |  (X_z < -cfg['iqr_factor']*iqr_val)] = np.nan

                X_z = interpol(X_z)
            try:
                X_len = len(data.T)
                pos_temp += X_len
                pos.append(pos_temp)
                X_train.append(X_z)
            except Exception as e:
                print(f"Error occurred while processing {file}. Skipping this file.")
                print(e)
                continue
        except Exception as e:
            print(f"Error occurred while processing {file}. Skipping this file.")
            print(e)
            continue
    
    X = np.concatenate(X_train, axis=0)
    # X_std = np.std(X)
    
    detect_anchors = np.std(X.T, axis=1)
    sort_anchors = np.sort(detect_anchors)
    if sort_anchors[0] == sort_anchors[1]:
        anchors = np.where(detect_anchors == sort_anchors[0])[0]
        anchor_1_temp = anchors[0]
        anchor_2_temp = anchors[1]
        
    else:
        anchor_1_temp = int(np.where(detect_anchors == sort_anchors[0])[0])
        anchor_2_temp = int(np.where(detect_anchors == sort_anchors[1])[0])
    
    if anchor_1_temp > anchor_2_temp:
        anchor_1 = anchor_1_temp
        anchor_2 = anchor_2_temp
        
    else:
        anchor_1 = anchor_2_temp
        anchor_2 = anchor_1_temp
    
    X = np.delete(X, anchor_1, 1)
    X = np.delete(X, anchor_2, 1)
    
    X = X.T
    
    if savgol_filter:
        X_med = scipy.signal.savgol_filter(X, cfg['savgol_length'], cfg['savgol_order'])
    else:
        X_med = X
        
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)
    
    z_test = X_med[:,:test]
    z_train = X_med[:,test:]
      
    if check_parameter == True:
        plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med) # , anchor_1, anchor_2)
        
    else:        
        #save numpy arrays the the test/train info:
        np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
        np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)
        
        for i, file in enumerate(files):
            try:
                np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])
            except Exception as e:
                print(f"Error occurred while saving {file}. Skipping this file.")
                print(e)
                continue
        
        print('Lenght of train data: %d' %len(z_train.T))
        print('Lenght of test data: %d' %len(z_test.T))
    
def traindata_aligned_fractional(cfg, files, testfraction, num_features, savgol_filter, check_parameter, data_fraction):
    rnd_frames = []
    last_frames = []
    file_names = []
    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    
    if check_parameter == True:
        X_true = []
        files = [files[0]]
        
    for file in files:
        try: 
            print("z-scoring of file %s" %file)
            path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
            
            # 1. File Loading
            try:
                file_names.append(file)
                data = np.load(path_to_file)
            except Exception as e:
                print(f"Error occurred while loading {file}. Skipping this file.")
                print(e)
                continue
            
            if data_fraction != 1.0:
                # 1.5 Get the fractional data
                try:
                    # Get the number of frames in the data
                    frames = len(data.T)
                    # Get the number of frames to keep
                    kept_frames = int(frames * data_fraction)
                    
                    # Set the min and max frames to keep
                    min_frame = 0
                    max_frame = int(frames - (frames * data_fraction))
                    
                    # Get a random frame to start at (ensures that the data is not biased in the time domain)
                    rnd_frame = np.random.randint(min_frame, max_frame)
                    
                    rnd_frames.append(rnd_frame)
                    
                    # Get the last frame to keep
                    last_frame = rnd_frame + kept_frames
                    
                    last_frames.append(last_frame)
                    
                    # Reduce the data to the fractional data
                    data = data[:, rnd_frame:last_frame]
                    
                except Exception as e:
                    print(f"Error occurred while getting fractional data for {file}. Skipping this file.")
                    print(e)
                    continue
            else:
                rnd_frame = 0
                last_frame = len(data.T)
                    
                
                
            # 2. Z-Scoring
            try:
                X_mean = np.mean(data,axis=None)
                X_std = np.std(data, axis=None)
                X_z = (data.T - X_mean) / X_std
            except Exception as e:
                print(f"Error occurred while z-scoring {file}. Skipping this file.")
                print(e)
            
            # Introducing artificial error spikes
            # rang = [1.5, 2, 2.5, 3, 3.5, 3, 3, 2.5, 2, 1.5]
            # for i in range(num_frames):
            #     if i % 300 == 0:
            #         rnd = np.random.choice(12,2)
            #         for j in range(10):
            #             X_z[i+j, rnd[0]] = X_z[i+j, rnd[0]] * rang[j]
            #             X_z[i+j, rnd[1]] = X_z[i+j, rnd[1]] * rang[j]
                    
            if check_parameter == True:
                X_z_copy = X_z.copy()
                X_true.append(X_z_copy)
                
            if cfg['robust'] == True:
                iqr_val = iqr(X_z)
                print("IQR value: %.2f, IQR cutoff: %.2f" %(iqr_val, cfg['iqr_factor']*iqr_val))
                X_z[(X_z > cfg['iqr_factor']*iqr_val) |  (X_z < -cfg['iqr_factor']*iqr_val)] = np.nan

                X_z = interpol(X_z)
            try:
                X_len = len(data.T)
                pos_temp += X_len
                pos.append(pos_temp)
                X_train.append(X_z)
            except Exception as e:
                print(f"Error occurred while processing {file}. Skipping this file.")
                print(e)
                continue
        except Exception as e:
            print(f"Error occurred while processing {file}. Skipping this file.")
            print(e)
            continue
    
    if data_fraction != 1.0:
         # Save last_frame and rnd_frame to a dataframe
        df = pd.DataFrame({'file_name': file_names, 'rnd_frame': rnd_frames, 'last_frame': last_frames})
        df.to_csv(os.path.join(cfg['project_path'],"data", 'PE-seq-fractional-information.csv'))

    
    X = np.concatenate(X_train, axis=0)
    # X_std = np.std(X)
    
    detect_anchors = np.std(X.T, axis=1)
    sort_anchors = np.sort(detect_anchors)
    if sort_anchors[0] == sort_anchors[1]:
        anchors = np.where(detect_anchors == sort_anchors[0])[0]
        anchor_1_temp = anchors[0]
        anchor_2_temp = anchors[1]
        
    else:
        anchor_1_temp = int(np.where(detect_anchors == sort_anchors[0])[0])
        anchor_2_temp = int(np.where(detect_anchors == sort_anchors[1])[0])
    
    if anchor_1_temp > anchor_2_temp:
        anchor_1 = anchor_1_temp
        anchor_2 = anchor_2_temp
        
    else:
        anchor_1 = anchor_2_temp
        anchor_2 = anchor_1_temp
    
    X = np.delete(X, anchor_1, 1)
    X = np.delete(X, anchor_2, 1)
    
    X = X.T
    
    if savgol_filter:
        X_med = scipy.signal.savgol_filter(X, cfg['savgol_length'], cfg['savgol_order'])
    else:
        X_med = X
        
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)
    
    z_test = X_med[:,:test]
    z_train = X_med[:,test:]
      
    if check_parameter == True:
        plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med, anchor_1, anchor_2)
        
    else:        
        #save numpy arrays the the test/train info:
        np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
        np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)
        
        # Have the user choose a suffix to denote this training set
        suffix = input("Please enter a suffix to denote this training set: ")

        # Get the current date
        today = date.today()

        # Update the config file with the suffix, adding the suffix to load_data: -PE-seq-clean_{today}_{suffix}.npy after wiping the text that was there before
        cfg['load_data'] = cfg['load_data'].split('_')[0] + f"_PE-seq-clean_{today}_{suffix}.npy"
        
        for i, file in enumerate(files):
            try:
                #np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])
                np.save(os.path.join(cfg['project_path'],"data", file, f"{file}-PE-seq-clean_{today}_{suffix}.npy"), X_med[:,pos[i]:pos[i+1]])
            except Exception as e:
                print(f"Error occurred while saving {file}. Skipping this file.")
                print(e)
                continue
        
        print('Lenght of train data: %d' %len(z_train.T))
        print('Lenght of test data: %d' %len(z_test.T))
    
    

def traindata_fixed(cfg, files, testfraction, num_features, savgol_filter, check_parameter):
    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    
    if check_parameter == True:
        X_true = []
        rnd_file = np.random.choice(len(files))
        files = [files[0]]
        
    for file in files:
        print("z-scoring of file %s" %file)
        path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        data = np.load(path_to_file)
        X_mean = np.mean(data,axis=None)
        X_std = np.std(data, axis=None)
        X_z = (data.T - X_mean) / X_std
        
        if check_parameter == True:
            X_z_copy = X_z.copy()
            X_true.append(X_z_copy)
        
        if cfg['robust'] == True:
            iqr_val = iqr(X_z)
            print("IQR value: %.2f, IQR cutoff: %.2f" %(iqr_val, cfg['iqr_factor']*iqr_val))
            
            # Create a mask for values that are outside the acceptable range
            mask = np.abs(X_z) > cfg['iqr_factor']*iqr_val
            
            # Set those values to NaN
            X_z[mask] = np.nan
            
            # Interpolate NaN values
            X_z = np.apply_along_axis(interpol, axis=1, arr=X_z)
        
        X_len = len(data.T)
        pos_temp += X_len
        pos.append(pos_temp)
        X_train.append(X_z)
    
    X = np.concatenate(X_train, axis=0).T

    if savgol_filter:
        X_med = scipy.signal.savgol_filter(X, cfg['savgol_length'], cfg['savgol_order'])   
    else:
        X_med = X
        
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)
    
    z_test =X_med[:,:test]
    z_train = X_med[:,test:]
    
    if check_parameter == True:
        plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med)
        
    else:
        #save numpy arrays the the test/train info:
        np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
        np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)
        
        # Have the user choose a suffix to denote this training set
        suffix = input("Please enter a suffix to denote this training set: ")

        # Get the current date
        from datetime import date
        today = date.today()

        # Update the config file with the suffix, adding the suffix to load_data: -PE-seq-clean_{suffix}.npy after wiping the text that was there before
        cfg['load_data'] = cfg['load_data'].split('_')[0] + f"_PE-seq-clean_{today}_{suffix}.npy"

        for i, file in enumerate(files):
            #np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])
            np.save(os.path.join(cfg['project_path'],"data", file, f"{file}-PE-seq-clean_{today}_{suffix}.npy"), X_med[:,pos[i]:pos[i+1]])
        
            
        print('Lenght of train data: %d' %len(z_train.T))
        print('Lenght of test data: %d' %len(z_test.T))

def clean_input(input_str):
    # Perform all cleaning operations in a single function
    input_str = input_str.strip()
    input_str = os.path.splitext(input_str)[0]
    input_str = input_str.replace("'", "")
    input_str = input_str.replace(" ", "")
    input_str = input_str.replace("\n", "")
    input_str = input_str.replace("[", "")
    input_str = input_str.replace("]", "")
    return input_str

def create_trainset(config, check_parameter=False, data_fraction=None):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    fixed = cfg['egocentric_data']
    
    if not os.path.exists(os.path.join(cfg['project_path'],'data','train',"")):
        os.mkdir(os.path.join(cfg['project_path'],'data','train',""))

    files = []
    if cfg['all_data'] == 'No':
        for file in cfg['video_sets']:
            use_list = input("Do you have a list of videos you want to use for training? yes/no: ")
            if use_list == 'yes':
                files_input = input("Please enter the list of videos you want to use for training: ")
                files = [clean_input(f) for f in files_input.split(',')]
                break
            elif use_list == 'no':
                use_file = input("Do you want to train on " + file + "? yes/no: ")
                if use_file == 'yes':
                    files.append(file)
                if use_file == 'no':
                    continue
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
    else:
        for file in cfg['video_sets']:
            files.append(file)

    print("Creating training dataset...")
    if cfg['robust'] == True:
        print("Using robust setting to eliminate outliers! IQR factor: %d" %cfg['iqr_factor'])
        
    if fixed == False:
        if data_fraction == None:
            print("Creating trainset from the vame.egocentrical_alignment() output ")
            traindata_aligned(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'], check_parameter)
        elif data_fraction != None:
            print("Creating trainset from the vame.egocentrical_alignment() output ")
            traindata_aligned_fractional(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'], check_parameter, data_fraction)
    else:
        print("Creating trainset from the vame.csv_to_numpy() output ")
        traindata_fixed(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'], check_parameter)
    
    if check_parameter == False:
        print("A training and test set has been created. Next step: vame.train_model()")
