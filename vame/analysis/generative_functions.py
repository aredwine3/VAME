"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

# Set the Matplotlib backend based on the environment.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use this backend for headless environments (e.g., Google Colab, some remote servers)
else:
    matplotlib.use('Qt5Agg')  # Use this backend for environments with a display server

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from vame.model.rnn_model import RNN_VAE
from vame.util.auxiliary import read_config


def random_generative_samples_motif(cfg, model, latent_vector,labels,n_cluster, path):
    # Latent sampling and generative model
    time_window = cfg['time_window']
    sampleName = path.split('/')[-4]
    for j in range(n_cluster):
        
        inds=np.where(labels==j)
        motif_latents=latent_vector[inds[0],:]
        gm = GaussianMixture(n_components=10).fit(motif_latents)
        
        # draw sample from GMM
        density_sample = gm.sample(10)
        
        # generate image via model decoder
        tensor_sample = torch.from_numpy(density_sample[0]).type('torch.FloatTensor').cuda()
        decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
        decoder_inputs = decoder_inputs.permute(0,2,1)
        
        image_sample = model.decoder(decoder_inputs, tensor_sample)
        recon_sample = image_sample.cpu().detach().numpy()
        
      
        fig, axs = plt.subplots(2,5)
        for i in range(5):
            axs[0,i].plot(recon_sample[i,...])
            axs[1,i].plot(recon_sample[i+5,...])
        plt.suptitle('Generated samples for motif '+str(j))
        fig.savefig(os.path.join(path, sampleName+'_GeneratedSamples_Motif'+str(j)+'.png'))
        plt.close('all')

def random_generative_samples(cfg, model, latent_vector):
    # Latent sampling and generative model
    time_window = cfg['time_window']
    gm = GaussianMixture(n_components=10).fit(latent_vector)
    
    # draw sample from GMM
    density_sample = gm.sample(10)
    
    # generate image via model decoder
    tensor_sample = torch.from_numpy(density_sample[0]).type('torch.FloatTensor').cuda()
    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0,2,1)
    
    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()
    
    fig, axs = plt.subplots(2,5)
    for i in range(5):
        axs[0,i].plot(recon_sample[i,...])
        axs[1,i].plot(recon_sample[i+5,...])
    plt.suptitle('Generated samples')
    

def random_reconstruction_samples(cfg, model, latent_vector):
    # random samples for reconstruction
    time_window = cfg['time_window']
    
    rnd = np.random.choice(latent_vector.shape[0], 10)
    tensor_sample = torch.from_numpy(latent_vector[rnd]).type('torch.FloatTensor').cuda()
    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0,2,1)
    
    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()
    
    fig, axs = plt.subplots(2,5)
    for i in range(5):
        axs[0,i].plot(recon_sample[i,...])
        axs[1,i].plot(recon_sample[i+5,...])
    plt.suptitle('Reconstructed samples')


def visualize_cluster_center(cfg, model, cluster_center):
    #Cluster Center
    time_window = cfg['time_window']
    animal_centers = cluster_center
    
    tensor_sample = torch.from_numpy(animal_centers).type('torch.FloatTensor').cuda()
    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0,2,1)
    
    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()
    
    num = animal_centers.shape[0]
    b = int(np.ceil(num / 5))
        
    fig, axs = plt.subplots(5,b)
    idx = 0
    for k in range(5):
        for i in range(b):
            axs[k,i].plot(recon_sample[idx,...])
            axs[k,i].set_title("Cluster %d" %idx)
            idx +=1
        

def load_model(cfg, model_name):
    # load Model
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    
    NUM_FEATURES = cfg['num_features']
    NUM_FEATURES = NUM_FEATURES - 2
    
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']
     
    print('Load model... ')
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    print(f"Using {device} device")
    # Check if cuda is available

    model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                            hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                            dropout_rec, dropout_pred, softplus).to(device)
    
    print("Model parameters:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',model_name+'_'+cfg['Project']+'.pkl'), map_location=device))
    model.eval()
    
    return model


def generative_model(config, mode="sampling"):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    
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
        
    
    model = load_model(cfg, model_name)
    
    for file in files:
        path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")

        if mode == "sampling":
            latent_vector = np.load(os.path.join(path_to_file,'latent_vector_'+file+'.npy'))
            random_generative_samples(cfg, model, latent_vector)
        
        if mode == "reconstruction":
            latent_vector = np.load(os.path.join(path_to_file,'latent_vector_'+file+'.npy'))
            random_reconstruction_samples(cfg, model, latent_vector)
            
        if mode == "centers":
            cluster_center = np.load(os.path.join(path_to_file,'cluster_center_'+file+'.npy'))
            visualize_cluster_center(cfg, model, cluster_center)

        if mode == "motifs":
            latent_vector = np.load(os.path.join(path_to_file,'latent_vector_'+file+'.npy'))
            labels = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+file+'.npy'))
            random_generative_samples_motif(cfg, model, latent_vector,labels,n_cluster, path_to_file)            

    









