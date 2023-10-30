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
from pathlib import Path
import glob
import shutil
import torch


from hmmlearn import hmm
from sklearn.cluster import KMeans
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE


def load_model(cfg, model_name, fixed):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pass
    else:
        torch.device("cpu")
    
    # load Model
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    if fixed == False:
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


def embedd_latent_vectors(cfg, files, model, fixed):
    project_path = cfg['project_path']
    temp_win = cfg['time_window']
    num_features = cfg['num_features']
    if fixed == False:
        num_features = num_features - 2
        
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pass
    else:
        torch.device("cpu")
        
    latent_vector_files = [] 

    # Get the text following "load_data:" in the config file
    load_data = cfg['load_data']

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

def load_latent_vectors(cfg, files):
    project_path = cfg['project_path']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    
    latent_vectors=[]
    for file in files:
        resultPath = os.path.join(cfg['project_path'],"results",file,model_name)                    
        latent_vec = glob.glob(os.path.join(resultPath,parameterization+'-*','latent_vector_'+file+'.npy'))[0]
        vec = np.load(latent_vec)
        latent_vectors.append(vec)
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
                usage_list.insert(i,0)
        motif_usage = np.array(usage_list)
    else:
        motif_usage = motif_usage[1]
    
    return motif_usage


def same_parameterization(cfg, files, latent_vector_files, states, parameterization, hmm_iters=200):
    random_state = cfg['random_state_kmeans']
    model_name = cfg['model_name']
    n_init = cfg['n_init_kmeans']
    n_cluster=cfg['n_cluster']
    labels = []
    cluster_centers = []
    motif_usages = []
    
    latent_vector_cat = np.concatenate(latent_vector_files, axis=0)
        
    if parameterization == "kmeans":
        print("Using kmeans as parameterization!")
        kmeans = KMeans(init='k-means++', n_clusters=states, random_state=random_state, n_init=n_init).fit(latent_vector_cat)
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(latent_vector_cat)
        
    elif parameterization == "hmm":
        if cfg['hmm_trained'] == False:
            print("Using a HMM as parameterization!")
            #hmm_model = hmm.GaussianHMM(n_components=states, covariance_type="full", n_iter=hmm_iters, verbose=True)
            #hmm_model.fit(latent_vector_cat)
            #label = hmm_model.predict(latent_vector_cat)
            
            
            # Number of states and features
            n_states = states  # Make sure this is the same as n_components in GaussianHMM
            n_features = len(latent_vector_cat[0])
            
            initial_means = [np.zeros(n_features) for _ in range(n_states)]  # Replace with your own initial means
            initial_covs = [np.identity(n_features) for _ in range(n_states)]  # Replace with your own initial covariances


            # Initialize distributions
            # Here, you might want to initialize based on some statistics of your data
            distributions = []
            for i in range(states):
                state_distribution = Normal(means=initial_means[i], covs=initial_covs[i], covariance_type='full')
                distributions.append(state_distribution)


            # Initialize transition matrix
            # Again, consider initializing based on data or domain knowledge
            edges = np.full((n_states, n_states), 1.0 / n_states)

            # Initialize start and end probabilities
            starts = np.full(n_states, 1.0 / n_states)
            ends = np.full(n_states, 1.0 / n_states)

            model = DenseHMM(distributions=distributions, edges=edges, starts=starts, ends=ends, max_iter=hmm_iters, verbose=True, check_data=True)
            
            model.fit(latent_vector_cat)
            

            label = model.predict(latent_vector_cat)
            
            
            save_data = os.path.join(cfg['project_path'], "results")
            #with open(os.path.join(save_data, "hmm_trained_ncluster"+str(states)+".pkl"), "wb") as file: pickle.dump(hmm_model, file)
            with open(os.path.join(save_data, f"hmm_trained_ncluster{states}_{model_name}.pkl"), "wb") as file:
                pickle.dump(hmm_model, file)
        else:
            print("Using a pretrained HMM as parameterization!")
            save_data = os.path.join(cfg['project_path'], "results")
            #with open(os.path.join(save_data, "hmm_trained_ncluster"+str(states)+".pkl"), "rb") as file:
            #    hmm_model = pickle.load(file)
            with open(os.path.join(save_data, f"hmm_trained_ncluster{states}_{model_name}.pkl"), "rb") as file:
                pickle.dump(hmm_model, file)
                
            label = hmm_model.predict(latent_vector_cat)
        
    idx = 0
    for i, file in enumerate(files):
        file_len = latent_vector_files[i].shape[0]
        labels.append(label[idx:idx+file_len])
        if parameterization == "kmeans":
            cluster_centers.append(clust_center)
        
        motif_usage = get_motif_usage(label[idx:idx+file_len], n_cluster)
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


def pose_segmentation(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    fixed = cfg['egocentric_data']
    parameterization = cfg['parameterization']
    if parameterization=='hmm':
        hmm_iters=cfg['hmm_iters']
    
    print('Pose segmentation for VAME model: %s \n' %model_name)
    
    if legacy == True:
        from segment_behavior import behavior_segmentation
        behavior_segmentation(config, model_name=model_name, cluster_method='kmeans', n_cluster=n_cluster)
        
    else:
        ind_param = cfg['individual_parameterization']
        
        for folders in cfg['video_sets']:
            if not os.path.exists(os.path.join(cfg['project_path'],"results",folders,model_name,"")):
                os.mkdir(os.path.join(cfg['project_path'],"results",folders,model_name,""))
    
        files = []
        if cfg['all_data'] == 'No':
            all_flag = input("Do you want to qunatify your entire dataset? \n"
                              "If you only want to use a specific dataset type filename: \n"
                              "yes/no/filename ")
            file = all_flag
            
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
        # files.append("mouse-3-1")
        # file="mouse-3-1"
    
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("Using CUDA")
            print('GPU active:',torch.cuda.is_available())
            print('GPU used:',torch.cuda.get_device_name(0))
        else:
            print("CUDA is not working! Attempting to use the CPU...")
            torch.device("cpu")

        for file in files:        
            if glob.glob(os.path.join(cfg['project_path'], "results", files[-1], model_name, '*','latent_vector*')):
                resultPath = os.path.join(cfg['project_path'],"results",file,model_name)                    
                latent_vec = glob.glob(os.path.join(resultPath,'*','latent_vector_'+file+'.npy'))
                latent_vec_ncluster = latent_vec[0].split('/')[-2].split('-')[1]                    
                if not os.path.exists(os.path.join(resultPath,parameterization+'-'+str(n_cluster))):
                    os.mkdir(os.path.join(resultPath, parameterization+'-'+str(n_cluster)))
                    new = True
                    src=latent_vec[0]
                    dest=os.path.join(resultPath,parameterization+'-'+str(n_cluster))
                    if not glob.glob(os.path.join(dest, 'latent_vector*')):
                        print("Latent vector found for "+str(latent_vec_ncluster)+" clusters. Copying file.")
                        shutil.copy(src, dest)
                  #      print("Copied latent vector for " + file + " from n_cluster" + str(latent_vec_ncluster) + " to " + str(n_cluster))
                else:          
                    print('\n'
                          'For model %s a latent vector embedding already exists. \n' 
                          'Parameterization of latent vector with %d k-Means cluster' %(model_name, n_cluster))
                    

        if not os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name, parameterization+'-'+str(n_cluster))):
            new = True
            # print("Hello1")
            model = load_model(cfg, model_name, fixed)
            latent_vectors = embedd_latent_vectors(cfg, files, model, fixed)

        else:
            print("Loading previously calculated latent vectors")
            latent_vectors = load_latent_vectors(cfg, files)
            new = True
            
#        if ind_param == False:
#            print("For all animals the same parameterization of latent vectors is applied for %d cluster" %n_cluster)
#            labels, cluster_center, motif_usages = same_parameterization(cfg, files, latent_vectors, n_cluster, parameterization, hmm_iters=hmm_iters)
#        else:
#            print("Individual parameterization of latent vectors for %d cluster" %n_cluster)
#            labels, cluster_center, motif_usages = individual_parameterization(cfg, files, latent_vectors, n_cluster)
                    
        if os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name, parameterization+'-'+str(n_cluster))):
            flag = input('WARNING: A parameterization for the chosen cluster size of the model may already exist! \n'
                        'Do you want to continue? If hmm_trained in your config file is "false", a new parameterization will be computed.\n'
                        'If hmm_trained is "true", the previous parameterization will be loaded (yes/no) ').lower()
        else:
            flag = 'yes'
        
        if flag == 'yes':
            new = True
     #       latent_vectors = []
     #       for file in files:
     #           path_to_latent_vector = os.path.join(cfg['project_path'],"results",file,model_name, parameterization+'-'+str(n_cluster),"")
     #           latent_vector = np.load(os.path.join(path_to_latent_vector,'latent_vector_'+file+'.npy'))
     #           latent_vectors.append(latent_vector)
                
            if ind_param == False:
                print("For all animals the same parameterization of latent vectors is applied for %d cluster" %n_cluster)
                labels, cluster_center, motif_usages = same_parameterization(cfg, files, latent_vectors, n_cluster, parameterization, hmm_iters=hmm_iters)
            else:
                print("Individual parameterization of latent vectors for %d cluster" %n_cluster)
                labels, cluster_center, motif_usages = individual_parameterization(cfg, files, latent_vectors, n_cluster)

   #         else:
   #             print('No new parameterization has been calculated.')
   #             new = False
                
        # print("Hello2")
        if new == True:
            for idx, file in enumerate(files):
                print(os.path.join(cfg['project_path'],"results",file,"",model_name,parameterization+'-'+str(n_cluster)))
                if not os.path.exists(os.path.join(cfg['project_path'],"results",file,model_name,parameterization+'-'+str(n_cluster))):                    
                    try:
                        os.mkdir(os.path.join(cfg['project_path'],"results",file,"",model_name,parameterization+'-'+str(n_cluster)))
                    except OSError as error:
                        print(error)   
                    
                save_data = os.path.join(cfg['project_path'],"results",file,model_name,parameterization+'-'+str(n_cluster))
                np.save(os.path.join(save_data,str(n_cluster)+'_km_label_'+file), labels[idx])
                if parameterization=="kmeans":
                    np.save(os.path.join(save_data,'cluster_center_'+file), cluster_center[idx])
                np.save(os.path.join(save_data,'latent_vector_'+file), latent_vectors[idx])
                np.save(os.path.join(save_data,'motif_usage_'+file), motif_usages[idx])
    
        
            print("You succesfully extracted motifs with VAME! From here, you can proceed running vame.motif_videos() ")
                  # "to get an idea of the behavior captured by VAME. This will leave you with short snippets of certain movements."
                  # "To get the full picture of the spatiotemporal dynamic we recommend applying our community approach afterwards.")
            
