#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os


class SEQUENCE_DATASET(Dataset):
    """
    This class is a custom dataset class for handling sequence data. It inherits from PyTorch's Dataset class.
    """
    def __init__(self,path_to_file,data,train,temporal_window):
        """
        Initialize the dataset object.
        
        Parameters:
        path_to_file (str): The path to the file containing the data.
        data (str): The name of the data file.
        train (bool): A flag indicating whether the data is for training or testing.
        temporal_window (int): The size of the temporal window for the sequence.
        """
        self.temporal_window = temporal_window        
        self.X = np.load(path_to_file+data)  # Load the data from the file
        
        # Transpose the data if necessary
        if self.X.shape[0] > self.X.shape[1]:
            self.X=self.X.T
            
        # Get the number of data points
        self.data_points = len(self.X[0,:])
        
        # Compute the mean and standard deviation of the data if they do not exist
        if train and not os.path.exists(os.path.join(path_to_file,'seq_mean.npy')):
            print("Compute mean and std for temporal dataset.")
            self.mean = np.mean(self.X)
            self.std = np.std(self.X)
            np.save(path_to_file+'seq_mean.npy', self.mean)
            np.save(path_to_file+'seq_std.npy', self.std)
        else:
            # Load the mean and standard deviation from the files
            self.mean = np.load(path_to_file+'seq_mean.npy')
            self.std = np.load(path_to_file+'seq_std.npy')
        
        # Print the number of data points
        if train:
            print('Initialize train data. Datapoints %d' %self.data_points)
        else:
            print('Initialize test data. Datapoints %d' %self.data_points)
        
    def __len__(self):        
        """
        Return the number of data points.
        """
        return self.data_points

    def __getitem__(self, index):
        """
        Get a sequence of data points for a given index.
        
        Parameters:
        index (int): The index of the sequence.
        
        Returns:
        torch.Tensor: The sequence of data points.
        """
        temp_window = self.temporal_window
        
        nf = self.data_points
        
        start = np.random.choice(nf-temp_window)

        end = start+temp_window
        
        # Get the sequence of data points
        sequence = self.X[:,start:end]  

        # Normalize the sequence
        sequence = (sequence-self.mean)/self.std
            
        # Convert the sequence to a PyTorch tensor and return it
        return torch.from_numpy(sequence)
    
    
    
    
    
