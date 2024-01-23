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
import tqdm
import torch
import pickle
import numpy as np
import glob
import shutil
from icecream import ic
import gc
import math
import time
from hmmlearn import hmm
from sklearn.cluster import KMeans
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal


from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE

def load_model(cfg, model_name, fixed):
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        pass
    else:
        torch.device("cpu")
        
        # load Model
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    if fixed is False:
        NUM_FEATURES = NUM_FEATURES - 2
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']
    
    
    if use_gpu:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
            hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
            dropout_rec, dropout_pred, softplus).cuda()
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
            hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
            dropout_rec, dropout_pred, softplus).to()
        
    model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',model_name+'_'+cfg['Project']+'.pkl')))
    model.eval()
    
    
    return model


def embedd_latent_vectors_working(cfg, files, model, fixed):
    project_path = cfg['project_path']
    temp_win = cfg['time_window']
    num_features = cfg['num_features']
    load_data = cfg['load_data']
    if fixed is False:
        num_features = num_features - 2
        
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pass
    else:
        torch.device("cpu")
        
    latent_vector_files = [] 
    
    # Get the text following "load_data:" in the config file
    
    
    
    for file in files:
        print('Embedding of latent vector for file %s' %file)
        data = np.load(os.path.join(project_path,'data',file,file+load_data+'.npy'))
        #data = np.load(os.path.join(project_path,'data',file,file+'-PE-seq-clean.npy'))
        latent_vector_list = []
        with torch.no_grad(): 
            for i in tqdm.tqdm(range(data.shape[1] - temp_win)):
                # for i in tqdm.tqdm(range(10000)):
                data_sample_np = data[:,i:temp_win+i].T
                data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features))
                
                if use_gpu:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type('torch.FloatTensor').cuda())
                else:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type('torch.FloatTensor').to())
                mu, _, _ = model.lmbda(h_n)
                
                latent_vector_list.append(mu.cpu().data.numpy())
                
        latent_vector = np.concatenate(latent_vector_list, axis=0)
        latent_vector_files.append(latent_vector)
        
    return latent_vector_files

def embedd_latent_vectors(cfg, files, model, fixed, new_latent_vectors = True):
    """Embed latent vectors for a list of files.
        
        This function runs the trained VAE model to embed each timestep 
        of the pose estimation data for the given files into the 
        latent space. The latent vectors are saved to disk and returned.
        
        Args:
            cfg: Config dictionary.
            files: List of file names.
            model: Trained VAE model. 
            fixed: Whether to use fixed_inputs in the model.
        
        Returns:
            latent_vector_files: List of embedded latent vectors 
                for each file.
        """
    project_path = cfg['project_path']
    temp_win = cfg['time_window']
    num_features = cfg['num_features']
    model_name = cfg['model_name']
    load_data = cfg['load_data']
    
    
    if not fixed:
        num_features -= 2
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    latent_vector_files = []
    
    for file in files:
        
        latent_vector_folder = os.path.join(project_path, 'latent_vectors', model_name, load_data)
        # Check if the folder exists, if not create it
        
        if not os.path.exists(latent_vector_folder):
            os.makedirs(latent_vector_folder)
            
            
        latent_vector_path = os.path.join(latent_vector_folder, f"{file}_latent_vectors.npy")
        
        print(f'Embedding of latent vector for file {file}')
        
        data_path = os.path.join(project_path, 'data', file, f"{file}{load_data}.npy")
        
        data = np.load(data_path)
        
        latent_vector_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(data.shape[1] - temp_win)):
                data_sample_np = data[:, i:temp_win + i].T
                data_sample = torch.tensor(data_sample_np, dtype=torch.float32).unsqueeze(0).to(device)
                h_n = model.encoder(data_sample)
                mu, _, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().numpy())
                
        latent_vector = np.concatenate(latent_vector_list, axis=0)
        
        np.save(latent_vector_path, latent_vector)
        
        latent_vector_files.append(latent_vector)
        
        
    return latent_vector_files



def load_latent_vectors_working(cfg, files):
    cfg['project_path']
    model_name = cfg['model_name']
    cfg['n_cluster']
    parameterization = cfg['parameterization']
    load_data = cfg['load_data']
    
    latent_vectors=[]
    for file in files:
        resultPath = os.path.join(cfg['project_path'],"results",file,model_name,load_data)                         
        latent_vec = glob.glob(os.path.join(resultPath,parameterization+'-*','latent_vector_'+file+'.npy'))[0]
        vec = np.load(latent_vec)
        latent_vectors.append(vec)
    return latent_vectors

def load_latent_vectors(cfg, files):
    project_path = cfg['project_path']
    model_name = cfg['model_name']
    cfg['n_cluster']
    cfg['parameterization']
    load_data = cfg['load_data']
    
    latent_vectors=[]
    for file in files:
        ic(file)
        latent_vector_folder = os.path.join(project_path, 'latent_vectors', model_name, load_data)
        latent_vector_path = os.path.join(latent_vector_folder, f"{file}_latent_vectors.npy")
        latent_vector = np.load(latent_vector_path)                       
        latent_vectors.append(latent_vector)
    return latent_vectors


def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def get_motif_usage(label, n_cluster):
    motif_usage = np.unique(label, return_counts=True)
    motif_usage=list(motif_usage)
    for i in range(n_cluster):
        if i < len(motif_usage[0]):
            if motif_usage[0][i]!=i:
                motif_usage[0] = np.insert(motif_usage[0], i, i, axis=0)
                motif_usage[1] = np.insert(motif_usage[1], i, 0, axis=0)
    motif_usage = tuple(motif_usage)
    cons = consecutive(motif_usage[0])
    if len(cons) != 1:
        used_motifs = list(motif_usage[0])
        usage_list = list(motif_usage[1])   
        for i in range(n_cluster):
            if i not in used_motifs:
                used_motifs.insert(i, i)
                print("Usage list:", usage_list)
                usage_list.insert(i,0)
        motif_usage = np.array(usage_list)
    else:
        motif_usage = motif_usage[1]
        
    return motif_usage


def pom_hmm_setup(states, latent_vector_cat, hmm_iters):
    # Number of states and features
    n_states = states  # Make sure this is the same as n_components in GaussianHMM
    n_features = len(latent_vector_cat[0])
    
    ic(n_features)
    
    initial_means = [np.zeros(n_features, dtype=np.float32) for _ in range(n_states)]
    initial_covs = [np.identity(n_features, dtype=np.float32) for _ in range(n_states)]
    
    # Initialize distributions
    distributions = []
    for i in range(states):
        state_distribution = Normal(means=initial_means[i], covs=initial_covs[i], covariance_type='full')
        distributions.append(state_distribution)
        
        
    edges = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float32)
    starts = np.full(n_states, 1.0 / n_states, dtype=np.float32)
    ends = np.full(n_states, 1.0 / n_states, dtype=np.float32)
    
    hmm_model = DenseHMM(distributions=distributions, edges=edges, starts=starts, ends=ends, max_iter=hmm_iters, verbose=True, check_data=True)
    
    return hmm_model


def pom_hmm_fit(hmm_model, latent_vector_cat, latent_vector_files):
    
    latent_vector_cat_tensor = torch.tensor(latent_vector_cat, dtype=torch.float32)
    
    # Convert latent_vector_files to a PyTorch tensor
    X = [torch.tensor(latent_vector_file, dtype=torch.float32) for latent_vector_file in latent_vector_files]
    
    
    torch.cuda.empty_cache()
    
    hmm_model.fit(X)
    
    return hmm_model, latent_vector_cat_tensor


def get_memory_per_set(latent_vector_files, hmm_model):
    
    latent_vector_cat = np.concatenate(latent_vector_files)  # This should have shape (1881360, 50)
    
    # Convert to a single numpy array
    latent_vector_cat = np.array(latent_vector_cat)
    
    device_idx = torch.cuda.current_device()
    device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
    
    # Determine how much memory is currently free on the users GPU
    memory_limit = torch.cuda.get_device_properties(device=device).total_memory
    current_memory = torch.cuda.memory_allocated(device=device)
    free_memory = memory_limit - current_memory
    
    #first_set_of_vectors = torch.tensor(latent_vector_files[0:1], dtype=torch.float32, device=device)
    
    
    #first_set_of_vectors = torch.tensor(first_set_of_vectors, dtype=torch.float32)
    
    # Measure initial memory usage
    initial_allocated = torch.cuda.memory_allocated(device=device)
    initial_cached = torch.cuda.memory_reserved(device=device)
    
    #hmm_model.summarize(first_set_of_vectors)
    
    hmm_model.summarize([latent_vector_cat])
    
    
    # Measure final memory usage
    final_allocated = torch.cuda.memory_allocated(device=device)
    final_cached = torch.cuda.memory_reserved(device=device)
    
    # Calculate the differences
    diff_allocated = final_allocated - initial_allocated
    diff_cached = final_cached - initial_cached
    
    # Convert the difference to GB
    diff_allocated = diff_allocated 
    diff_cached = diff_cached
    
    memory_per_set = diff_allocated + diff_cached
    
    if memory_per_set == 0:
        # Adjust the value of memory_per_set or handle the error appropriately
        
        #max_sets_per_batch = 60
        
        max_sets_per_batch = 140
    else:
        max_sets_per_batch = math.floor(free_memory // memory_per_set)
        
        # Calculate the total number of sets and batches
    n_total_sets = len(latent_vector_files)
    n_batches = math.ceil(n_total_sets / max_sets_per_batch)  # Use ceil to ensure all sets are covered
    
    return memory_per_set, max_sets_per_batch, n_batches



def pom_hmm_summarize_batch(hmm_model, hmm_iters, latent_vector_files, max_sets_per_batch, n_batches, tolerance=1e-4, patience=5):
    
    latent_vector_cat = np.concatenate(latent_vector_files)  # This should have shape (1881360, 50)
    ic(latent_vector_cat.shape)
    
    #latent_vector_files_array = np.array(latent_vector_files)
    
    logp, last_logp = None, None
    no_improvement_count = 0  # Initialize the counter for early stopping
    
    for iteration in range(1, hmm_iters + 1):
        start_time = time.time()
        #np.random.shuffle(latent_vector_files_array)
        
        #sequence_indices = np.arange(len(latent_vector_files_array))
        
        sequence_indices = np.arange(len(latent_vector_cat))
        
        np.random.shuffle(sequence_indices)  # Shuffle sequence indices
        
        batches = []
        for j in range(n_batches):
            start_idx = j * max_sets_per_batch
            end_idx = (j + 1) * max_sets_per_batch
            batch_indices = sequence_indices[start_idx:end_idx]
            
            batch = latent_vector_cat[batch_indices, :]
            
            #batch = latent_vector_files_array[batch_indices, :, :]
            #batch = latent_vector_files_array[start_idx:end_idx, :, :]
            
            batches.append(batch)
            
        batches = [torch.tensor(batch, dtype=torch.float32) for batch in batches]
        
        logp = 0
        for i, batch in enumerate(batches):
            logp += hmm_model.summarize(batch).sum()
            
        improvement = (logp - last_logp) if last_logp is not None else None
        duration = time.time() - start_time
        
        if improvement is not None:
            if improvement < tolerance:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  # reset counter if there's an improvement
        else:
            improvement = torch.tensor(float('inf'))  # Just to indicate the first iteration
            
        ic(iteration, duration, improvement.item(), logp.item())
        
        # Check early stopping condition
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {iteration} due to no improvement over {patience} consecutive iterations.")
            break
        
        last_logp = logp
        if no_improvement_count == 0:
            hmm_model.from_summaries()  # Update the model only if there was an improvement
            
            
    del latent_vector_cat
    
    return hmm_model


def pom_hmm_summarize_batch_concat(hmm_model, hmm_iters, latent_vector_files, tolerance=1e-4, patience=100):
    
    latent_vector_cat = np.concatenate(latent_vector_files, axis=0)  # This should have shape (1881360, 50)
    
    max_sets_per_batch = len(latent_vector_cat) // 4
    
    # Calculate the total number of sets and batches
    n_total_sets = len(latent_vector_cat)
    math.ceil(n_total_sets / max_sets_per_batch)  # Use ceil to ensure all sets are covered
    
    logp, last_logp = None, None
    no_improvement_count = 0  # Initialize the counter for early stopping
    
    batch_size = 78
    n_videos = len(latent_vector_files)
    
    for iteration in range(1, hmm_iters + 1):
        start_time = time.time()
        batches = []
        
        """
                sequence_indices = np.arange(len(latent_vector_cat))
                
                np.random.shuffle(sequence_indices)  # Shuffle sequence indices
                
                

                for j in range(n_batches):
                        start_idx = j * max_sets_per_batch
                        end_idx = (j + 1) * max_sets_per_batch
                        batch_indices = sequence_indices[start_idx:end_idx]
                        
                        batch = latent_vector_cat[batch_indices, :]
                                
                        batches.append(batch)
                
                batches = [torch.tensor(batch, dtype=torch.float32) for batch in batches]
                """
        
        for i in range(0, n_videos, batch_size):
            batch_indices = slice(i * 4020, (i + batch_size) * 4020)
            batch = latent_vector_cat[batch_indices, :]
            batches.append(torch.tensor(batch, dtype=torch.float32).view(-1, 4020, 50))
            
        logp = 0
        for i, batch in enumerate(batches):
            ic.disable()
            ic(batch)
            ic(batch.shape)
            
            logp += hmm_model.summarize(batch).sum()
            
            hmm_model.from_summaries()  # Update the model every batch....
            
        improvement = (logp - last_logp) if last_logp is not None else None
        duration = time.time() - start_time
        
        if improvement is not None:
            if improvement < tolerance:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  # reset counter if there's an improvement
        else:
            improvement = torch.tensor(float('inf'))  # Just to indicate the first iteration
            """
                # Check early stopping condition
                if no_improvement_count >= patience:
                        print(f"Early stopping at iteration {iteration} due to no improvement over {patience} consecutive iterations.")
                        break
                """
        last_logp = logp
        """
                if no_improvement_count == 0:
                        hmm_model.from_summaries()  # Update the model only if there was an improvement
                """
        
    del latent_vector_cat
    return hmm_model




def get_memory_per_video(latent_vector_files, hmm_model, safety_factor=0.8):
    
    gc.collect()
    latent_vector_cat = np.concatenate(latent_vector_files)  # This should have shape (1881360, 50)
    #latent_vector_files_array = np.array(latent_vector_files)
    #video_latent_vectors = (latent_vector_files_array[0, :, :]) # This should have shape (4020, 50)
    # Add an extra dimension to make it 3D
    #video_latent_vectors = np.expand_dims(video_latent_vectors, axis=0)  # New shape should be (1, 4020, 50)
    # Convert to PyTorch tensor if necessary
    #video_latent_vectors = torch.tensor(video_latent_vectors, dtype=torch.float32)
    latent_vector_cat_tensor = torch.tensor(latent_vector_cat, dtype=torch.float32)
    device_idx = torch.cuda.current_device()
    device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
    # Determine how much memory is currently free on the users GPU
    memory_limit = torch.cuda.get_device_properties(device=device).total_memory
    current_memory = torch.cuda.memory_allocated(device=device)
    free_memory = memory_limit - current_memory
    # Measure initial memory usage
    initial_allocated = torch.cuda.memory_allocated(device=device)
    initial_cached = torch.cuda.memory_reserved(device=device)
    #hmm_model.predict_proba(video_latent_vectors)
    hmm_model.predict_proba(latent_vector_cat_tensor)
    # Measure final memory usage
    final_allocated = torch.cuda.memory_allocated(device=device)
    final_cached = torch.cuda.memory_reserved(device=device)
    # Calculate the differences
    diff_allocated = final_allocated - initial_allocated
    diff_cached = final_cached - initial_cached
    # Convert the difference to GB
    diff_allocated = diff_allocated 
    diff_cached = diff_cached
    memory_per_set = diff_allocated + diff_cached
    if memory_per_set == 0:
        # Adjust the value of memory_per_set or handle the error appropriately
        max_sets_per_batch = 60
        #max_sets_per_batch = 140
    else:
        max_sets_per_batch = math.floor(free_memory*safety_factor // memory_per_set)
        # Calculate the total number of sets and batches
    n_total_sets = len(latent_vector_files)
    n_batches = math.ceil(n_total_sets / max_sets_per_batch)  # Use ceil to ensure all sets are covered
    # Clear the garbage
    #del video_latent_vectors
    del latent_vector_cat
    
    return memory_per_set, max_sets_per_batch, n_batches


def posterior_propbabilities_loop(latent_vector_files, hmm_model):
    
    
    memory_per_set, max_sets_per_batch, n_batches = get_memory_per_video(latent_vector_files, hmm_model)
    
    # Clear the cache
    torch.cuda.empty_cache()
    
    # Clear the garbage
    gc.collect()
    
    # Delete the latent_vector_files_array
    
    
    latent_vector_files_array = np.array(latent_vector_files)
    
    batches = []
    """
        for j in range(n_batches):
                start_idx = j * max_sets_per_batch
                end_idx = (j + 1) * max_sets_per_batch
                batch = latent_vector_files_array[start_idx:end_idx, :, :]
                batches.append(batch)
        
        # Convert each batch to a PyTorch tensor
        batches = [torch.tensor(batch, dtype=torch.float32) for batch in batches]
        
        posterior_ps = []
        
        for i, batch in enumerate(batches):
                ic(f"Processing batch {i+1} with shape {batch.shape}")
                
                # Capture the posterior probabilities for each batch
                posterior_ps.append(hmm_model.predict_proba(batch)) # append to list
        
        # Concatenate all the posterior probabilities into a single tensor
        posterior_probabilities = torch.cat(posterior_ps, dim=0)
        
        # covert posterior_ps to a PyTorch tensor
        posterior_ps = torch.tensor(posterior_ps, dtype=torch.float32)
        
        labels = torch.argmax(posterior_ps, dim=-1)
        """
    
    # Prepare batches as PyTorch tensors
    batches = [torch.tensor(latent_vector_files_array[start_idx:end_idx], dtype=torch.float32) 
        for j in range(n_batches)
        for start_idx, end_idx in [(j * max_sets_per_batch, (j + 1) * max_sets_per_batch)]]
    
    # Initialize list to store posterior probabilities
    posterior_probabilities = []
    
    # Loop through each batch and calculate probabilities
    for i, batch in enumerate(batches):
        ic(f"Processing batch {i+1} with shape {batch.shape}")
        
        # Get the posterior probabilities for the batch
        batch_posteriors = hmm_model.predict_proba(batch)
        posterior_probabilities.append(batch_posteriors)
        
        # Concatenate posterior probabilities from all batches
    all_posterior_probabilities = torch.cat(posterior_probabilities, dim=0)
    
    # Convert posterior probabilities to labels (most likely states)
    label = torch.argmax(all_posterior_probabilities, dim=-1)
    
    ic(all_posterior_probabilities)
    ic(all_posterior_probabilities.shape)
    ic(label)
    ic(label.shape)
    
    return posterior_probabilities, label

def same_parameterization(cfg, files, latent_vector_files, states, parameterization, run_type, hmm_iters, train_new_model):
    random_state = cfg['random_state_kmeans']
    model_name = cfg['model_name']
    n_init = cfg['n_init_kmeans']
    n_cluster=cfg['n_cluster']
    labels = []
    cluster_centers = []
    motif_usages = []
    hmm_iters = cfg['hmm_iters'] 
    
    label = None
    
    save_data = os.path.join(cfg['project_path'], "latent_vectors")
    
    latent_vector_cat = np.concatenate(latent_vector_files, axis=0) 
    
    if parameterization == "kmeans":
        print("Using kmeans as parameterization!")
        kmeans = KMeans(init='k-means++', n_clusters=states, random_state=random_state, verbose = 1, n_init=n_init).fit(latent_vector_cat)
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(latent_vector_cat)
        
    elif parameterization == "hmm":
        if cfg['hmm_trained'] is False:
            print("Using a HMM as parameterization!")
            if run_type == 'cpu':
                if train_new_model is True: 
                    hmm_model = hmm.GaussianHMM(n_components=states, covariance_type="full", n_iter=hmm_iters, verbose=True)
                    hmm_model.fit(latent_vector_cat)
                label = hmm_model.predict(latent_vector_cat)
                
            if run_type == 'gpu':
                
                hmm_model = pom_hmm_setup(states, latent_vector_cat, hmm_iters)
                hmm_model, latent_vector_cat_tensor = pom_hmm_fit(hmm_model, latent_vector_cat, latent_vector_files)                
                label = hmm_model.predict(latent_vector_cat_tensor)
                memory_per_set, max_sets_per_batch, n_batches = get_memory_per_set(latent_vector_files, hmm_model)
                hmm_model = pom_hmm_summarize_batch(hmm_model, hmm_iters, latent_vector_files, max_sets_per_batch, n_batches)
                
                hmm_model = pom_hmm_summarize_batch_concat(hmm_model, hmm_iters, latent_vector_files, tolerance=1e-4, patience=50)      
                posterior_probabilities, label = posterior_propbabilities_loop(latent_vector_files, hmm_model)
                label = posterior_probabilities.argmax(1)
                posterior_probabilities_np = posterior_probabilities.cpu().numpy()
                save_data = os.path.join(cfg['project_path'], "results")
                np.save(os.path.join(save_data, f"{model_name}_posterior_probabilities.npy"), posterior_probabilities_np)    
                # Clear the garbage
                del latent_vector_cat
                
                torch.cuda.empty_cache()
                
                save_data = os.path.join(cfg['project_path'], "results")
                
            save_data = os.path.join(cfg['project_path'], "results")
            
            with open(os.path.join(save_data, f"hmm_trained_ncluster{states}_{model_name}_{run_type}_{states}_{hmm_iters}.pkl"), "wb") as file:
                pickle.dump(hmm_model, file)
        else:
            print("Using a pretrained HMM as parameterization!")
            save_data = os.path.join(cfg['project_path'], "results")
            with open(os.path.join(save_data, f"hmm_trained_ncluster{states}_{model_name}_{run_type}_{states}_{hmm_iters}.pkl"), "rb") as file:             
                hmm_model = pickle.load(file)
                
            if run_type == 'cpu':
                ic("Getting lables from model")
                
                start_time = time.time()
                
                label = hmm_model.predict(latent_vector_cat)
                
                end_time = time.time()
                
                total_time = end_time - start_time
                
                ic(total_time)    
                # Convert label to NumPy array once before the loop
    if label is not None:
        if run_type == 'gpu':
            label_np = label.cpu().numpy()
        elif run_type == 'cpu':
            label_np = label
    else:
        print("Label is not set. Exiting.")
        return None, None, None
    idx = 0
    for i, file in enumerate(files):
        file_len = latent_vector_files[i].shape[0]
        labels.append(label_np[idx:idx+file_len])
        if parameterization == "kmeans":
            cluster_centers.append(clust_center)
            
        motif_usage = get_motif_usage(label_np[idx:idx+file_len], n_cluster)
        motif_usages.append(motif_usage)
        idx += file_len
        
    return labels, cluster_centers, motif_usages


def individual_parameterization(cfg, files, latent_vector_files, cluster):
    random_state = cfg['random_state_kmeans: ']
    n_init = cfg['n_init_kmeans']
    
    labels = []
    cluster_centers = []
    motif_usages = []
    for i, file in enumerate(files):
        print(file)
        kmeans = KMeans(init='k-means++', n_clusters=cluster, random_state=random_state, n_init=n_init).fit(latent_vector_files[i])
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(latent_vector_files[i])  
        motif_usage = get_motif_usage(label)
        motif_usages.append(motif_usage)
        labels.append(label)
        cluster_centers.append(clust_center)
        
    return labels, cluster_centers, motif_usages


def setup_computation_device():
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    
def input_warning():
    return input('WARNING: A parameterization for the chosen cluster size of the model may already exist! \n'
'Do you want to continue? If hmm_trained in your config file is "false", a new parameterization will be computed.\n'
'If hmm_trained is "true", the previous parameterization will be loaded (yes/no) ').lower()


def check_for_latent_vectors(files, cfg, model_name, load_data):
    n_cluster = cfg['n_cluster']
    paramaterization = cfg['parameterization']
    for file in files:        
        resultPath = os.path.join(cfg['project_path'], "results", file, model_name, load_data)
        if paramaterization == 'hmm':
            parameterized_path = os.path.join(resultPath, paramaterization + '-' + str(n_cluster )+ '-' + str(cfg['hmm_iters']))
        else:
            parameterized_path = os.path.join(resultPath, paramaterization + '-' + str(n_cluster))
        latent_vector_pattern = os.path.join(resultPath, '*', 'latent_vector_' + file + '.npy')
        latent_vectors = glob.glob(latent_vector_pattern)
        
        if latent_vectors:
            latent_vec_ncluster = latent_vectors[0].split('/')[-2].split('-')[1]
            if not os.path.exists(parameterized_path):
                os.makedirs(parameterized_path, exist_ok=True)
                print("Latent vector found for " + str(latent_vec_ncluster) + " clusters. Copying file.")
                shutil.copy(latent_vectors[0], parameterized_path)
            else:
                print('\n'
                    'For model %s a latent vector embedding already exists. \n' 
                    'Parameterization of latent vector with %d k-Means cluster' % (model_name, n_cluster))
        else:
            pass
            
def pose_segmentation(config, train_new_model = False, new_latent_vectors = True, run_type = 'cpu'):
    cfg = read_config(config)
    
    legacy = cfg['legacy']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    fixed = cfg['egocentric_data']
    parameterization = cfg['parameterization']
    load_data = cfg['load_data']
    hmm_iters = cfg.get('hmm_iters', 0)
    
    print('Pose segmentation for VAME model: %s \n' %model_name)
    
    if legacy:
        from segment_behavior import behavior_segmentation
        behavior_segmentation(config, model_name=model_name, cluster_method='kmeans', n_cluster=n_cluster)
        
    else:
        import vame.custom.ALR_helperFunctions as AlHf
        ind_param = cfg['individual_parameterization']
        
        for folders in cfg['video_sets']:
            if not os.path.exists(os.path.join(cfg['project_path'],"results",folders,model_name,"")):
                os.mkdir(os.path.join(cfg['project_path'],"results",folders,model_name,""))
                
        files = AlHf.get_files(config)
        check_for_latent_vectors(files, cfg, model_name, load_data)
        setup_computation_device()
        
        if train_new_model and parameterization == 'hmm':
            hmm_path = os.path.join(cfg['project_path'], "results", file, model_name, load_data, parameterization + '-' + str(n_cluster) + '-' + str(hmm_iters))
            if os.path.exists(hmm_path):
                input_warning()
                
        if new_latent_vectors:
            print("Embedding latent vectors")
            load_model(cfg, model_name, fixed)
            latent_vectors = embedd_latent_vectors(cfg, files, model, fixed)
        else:
            print("Loading previously calculated latent vectors")
            latent_vectors = load_latent_vectors(cfg, files)
            
        if ind_param is False:
            print("For all animals the same parameterization of latent vectors is applied for %d cluster" %n_cluster)
            labels, cluster_center, motif_usages = same_parameterization(cfg, files, latent_vectors, n_cluster, parameterization, run_type, hmm_iters, train_new_model)
        else:
            print("Individual parameterization of latent vectors for %d cluster" %n_cluster)
            labels, cluster_center, motif_usages = individual_parameterization(cfg, files, latent_vectors, n_cluster)
            
        for idx, file in enumerate(files):                
            if parameterization == 'hmm':
                save_data = os.path.join(cfg['project_path'], "results", file, model_name, load_data, parameterization+'-'+str(n_cluster)+'-'+str(hmm_iters))
                os.makedirs(save_data, exist_ok=True)
            elif parameterization == 'kmeans':
                save_data = os.path.join(cfg['project_path'],"results", file, model_name, load_data, parameterization+'-'+str(n_cluster))
                os.makedirs(save_data, exist_ok=True)
                
            np.save(os.path.join(save_data,str(n_cluster)+'_km_label_'+file), labels[idx])
            if parameterization=="kmeans":
                np.save(os.path.join(save_data,'cluster_center_'+file), cluster_center[idx])
            np.save(os.path.join(save_data,'latent_vector_'+file), latent_vectors[idx])
            np.save(os.path.join(save_data,'motif_usage_'+file), motif_usages[idx])
            
            
        print("You successfully extracted motifs with VAME! From here, you can proceed running vame.motif_videos() ")