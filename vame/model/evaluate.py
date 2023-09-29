#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
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

import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch.utils.data as Data

from torch.utils.data import DataLoader

from vame.util.auxiliary import read_config
from vame.model.rnn_vae import RNN_VAE
from vame.model.dataloader import SEQUENCE_DATASET

def set_device():
  # make sure torch uses cuda for GPU computing
  use_gpu = torch.cuda.is_available()
  use_mps = torch.backends.mps.is_available() and not use_gpu

  if use_gpu:
      device = torch.device("cuda")
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
      print("Using CUDA")
      print('GPU active:', torch.cuda.is_available())
      print('GPU used:', torch.cuda.get_device_name(0))
  elif use_mps:
      device = torch.device("mps")
      torch.set_default_tensor_type('torch.FloatTensor')
      print("Using MPS")
  else:
      device = torch.device("cpu")
      print("Using CPU")
      
  return device, use_gpu, use_mps

def to_cpu_numpy(tensor):
    return tensor.cpu().detach().numpy()

def plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name,
                        FUTURE_DECODER, FUTURE_STEPS, suffix=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Ensure the model is on the correct device
    model = model.to(device)

    x_iter = iter(test_loader)
    x = next(x_iter)
    x = x.to('cuda')
    x = x.permute(0,2,1)

    data = x[:,:seq_len_half,:].float().to(device)
    data_fut = x[:,seq_len_half:seq_len_half+FUTURE_STEPS,:].float().to(device)
    
    if FUTURE_DECODER:
        x_tilde, future, latent, mu, logvar = model(data)

        fut_orig = to_cpu_numpy(data_fut)
        fut = to_cpu_numpy(future)

    else:
        x_tilde, latent, mu, logvar = model(data)

    data_orig = to_cpu_numpy(data)
    data_tilde = to_cpu_numpy(x_tilde)

    if FUTURE_DECODER:
        fig, axs = plt.subplots(2, 5)
        fig.suptitle('Reconstruction [top] and future prediction [bottom] of input sequence')
        for i in range(5):
            axs[0,i].plot(data_orig[i,...], color='k', label='Sequence Data')
            axs[0,i].plot(data_tilde[i,...], color='r', linestyle='dashed', label='Sequence Reconstruction')

            axs[1,i].plot(fut_orig[i,...], color='k')
            axs[1,i].plot(fut[i,...], color='r', linestyle='dashed')
        axs[0,0].set(xlabel='time steps', ylabel='reconstruction')
        axs[1,0].set(xlabel='time steps', ylabel='predction')
        if not suffix:
            fig.savefig(os.path.join(filepath,"evaluate",'Future_Reconstruction.png'))
        elif suffix:
            fig.savefig(os.path.join(filepath,"evaluate",'Future_Reconstruction_'+suffix+'.png'))
        plt.close('all')

    else:
        fig, ax1 = plt.subplots(1, 5)
        for i in range(5):
        #    ax = ax1.flatten()
            fig.suptitle('Reconstruction of input sequence')
            ax1[i].plot(data_orig[i,...], color='k', label='Sequence Data')
            ax1[i].plot(data_tilde[i,...], color='r', linestyle='dashed', label='Sequence Reconstruction')
        fig.set_tight_layout(True)
        if not suffix:
            fig.savefig(os.path.join(filepath,'evaluate','Reconstruction_'+model_name+'.png'), bbox_inches='tight')
        elif suffix:
            fig.savefig(os.path.join(filepath,'evaluate','Reconstruction_'+model_name+'_'+suffix+'.png'), bbox_inches='tight')
        plt.close('all')

def plot_loss(cfg, filepath, model_name, suffix=None):
    basepath = os.path.join(cfg['project_path'],"model","model_losses")
    train_loss = np.load(os.path.join(basepath,'train_losses_'+model_name+'.npy'))
    test_loss = np.load(os.path.join(basepath,'test_losses_'+model_name+'.npy'))
    mse_loss_train = np.load(os.path.join(basepath,'mse_train_losses_'+model_name+'.npy'))
    mse_loss_test = np.load(os.path.join(basepath,'mse_test_losses_'+model_name+'.npy'))
#    km_loss = np.load(os.path.join(basepath,'kmeans_losses_'+model_name+'.npy'), allow_pickle=True)
    km_losses = np.load(os.path.join(basepath,'kmeans_losses_'+model_name+'.npy'),  allow_pickle=True)
    kl_loss = np.load(os.path.join(basepath,'kl_losses_'+model_name+'.npy'),  allow_pickle=True)
    fut_loss = np.load(os.path.join(basepath,'fut_losses_'+model_name+'.npy'), allow_pickle=True)

#    km_losses = []
#    for i in range(len(km_loss)):
#        km = km_loss[i].cpu().detach().numpy()
#        km_losses.append(km)

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Losses of our Model')
    ax1.set(xlabel='Epochs', ylabel='loss [log-scale]')
    ax1.set_yscale("log")
    ax1.plot(train_loss, label='Train-Loss')
    ax1.plot(test_loss, label='Test-Loss')
    ax1.plot(mse_loss_train, label='MSE-Train-Loss')
    ax1.plot(mse_loss_test, label='MSE-Test-Loss')
    ax1.plot(km_losses, label='KMeans-Loss')
    ax1.plot(kl_loss, label='KL-Loss')
    ax1.plot(fut_loss, label='Prediction-Loss')
    ax1.legend()
    #fig.savefig(filepath+'evaluate/'+'MSE-and-KL-Loss'+model_name+'.png')
    if not suffix:
        fig.savefig(os.path.join(filepath,"evaluate",'MSE-and-KL-Loss'+model_name+'.png'))
    elif suffix:
        fig.savefig(os.path.join(filepath,"evaluate",'MSE-and-KL-Loss'+model_name+'_'+suffix+'.png'))
    plt.close('all')


def eval_temporal(cfg, use_gpu, use_mps, model_name, fixed, snapshot=None, suffix=None):

    SEED = 19
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    if fixed == False:
        NUM_FEATURES = NUM_FEATURES - 2
    TEST_BATCH_SIZE = 64
    PROJECT_PATH = cfg['project_path']
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']

    filepath = os.path.join(cfg['project_path'],"model")

    device, use_gpu, use_mps = set_device()

    seq_len_half = int(TEMPORAL_WINDOW/2)
    if use_gpu:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus).cuda()
        if not snapshot:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl')))
        elif snapshot:
            model.load_state_dict(torch.load(snapshot))
    elif use_mps:
        torch.manual_seed(SEED)
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus).to(device)
        if not snapshot:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl')))
        elif snapshot:
            model.load_state_dict(torch.load(snapshot)) 
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus).to()
        if not snapshot:
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl'), map_location=torch.device('cpu')))
        elif snapshot:
            model.load_state_dict(torch.load(snapshot), map_location=torch.device('cpu'))
    
    model.eval() #toggle evaluation mode

    testset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='test_seq.npy', train=False, temporal_window=TEMPORAL_WINDOW)
    # Create a generator and place it on the right device
    generator = torch.Generator(device='cuda' if use_gpu else 'cpu')
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, generator=generator) # added pin_memory=True
    
    if not snapshot:
        plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name, FUTURE_DECODER, FUTURE_STEPS)#, suffix=suffix
    elif snapshot:
        plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name, FUTURE_DECODER, FUTURE_STEPS, suffix=suffix)#, 
    if use_gpu:
        if not suffix:
            plot_loss(cfg, filepath, model_name)
        elif suffix:
            plot_loss(cfg, filepath, model_name, suffix=suffix)
    elif use_mps:
        if not suffix:
            plot_loss(cfg, filepath, model_name)
        elif suffix:
            plot_loss(cfg, filepath, model_name, suffix=suffix)
    else:
        plot_loss(cfg, filepath, model_name)
        # pass #note, loading of losses needs to be adapted for CPU use #TODO

def evaluate_model(config, model_name, use_snapshots=False):#, suffix=None
    """
        Evaluation of testset.
        
    Parameters
    ----------
    config : str
        Path to config file.
    model_name : str
        name of model (same as in config.yaml)
    use_snapshots : bool
        Whether to plot for all snapshots or only the best model.
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    model_name = cfg['model_name']
    fixed = cfg['egocentric_data']
    

    if not os.path.exists(os.path.join(cfg['project_path'],"model","evaluate")):
        os.mkdir(os.path.join(cfg['project_path'],"model","evaluate"))

    device, use_gpu, use_mps = set_device()
    
    """
    use_gpu = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and not use_gpu

    if use_gpu:
        device = torch.device("cuda")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("Using CUDA")
        print('GPU active:', torch.cuda.is_available())
        print('GPU used:', torch.cuda.get_device_name(0))
    elif use_mps:
        device = torch.device("mps")
        torch.set_default_tensor_type('torch.FloatTensor')
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    """

    print("\n\nEvaluation of %s model. \n" %model_name)   

    if not use_snapshots:
        eval_temporal(cfg, use_gpu, use_mps, model_name, fixed)#suffix=suffix
    elif use_snapshots:
        snapshots=os.listdir(os.path.join(cfg['project_path'],'model','best_model','snapshots'))
        for snap in snapshots:
            fullpath = os.path.join(cfg['project_path'],"model","best_model","snapshots",snap)
            epoch=snap.split('_')[-1].strip('.pkl')
            eval_temporal(cfg, use_gpu, use_mps, model_name, fixed, snapshot=fullpath, suffix='snapshot'+str(epoch))
            eval_temporal(cfg, use_gpu, use_mps, model_name, fixed, suffix='bestModel')

    print("You can find the results of the evaluation in '/Your-VAME-Project-Apr30-2020/model/evaluate/' \n"
          "OPTIONS:\n"
          "- vame.pose_segmentation() to identify behavioral motifs.\n"
          "- re-run the model for further fine tuning. Check again with vame.evaluate_model()")
