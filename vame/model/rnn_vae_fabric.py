# Description: This file contains the training loop for the RNN-VAE model using the Fabric library.
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""
# Standard libraries
import os
import time
from pathlib import Path

# Third-party libraries
import vame
import wandb
import torch
import numpy as np
import logging
import psutil
import lightning as L
#from lightning.fabric import Fabric
import pandas as pd
from torch import nn
import torch.utils.data as Data
import torch.distributed as dist
import datetime
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.plugins.environments import MPIEnvironment

# Local application/library specific imports
from vame.util.auxiliary import read_config
from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE, RNN_VAE_LEGACY

# Warnings
import warnings

fabric = L.Fabric(
    accelerator="auto", 
    devices="auto", # number of GPUs
    strategy='ddp',
    #num_nodes=1,
    precision='32',
    #node_rank=node_rank,
    #master_address=master_addr,
    #main_port=master_port,
    #plugins=[SLURMEnvironment()]
    )


#fabric.launch()

device = fabric.device


# Ignore these specific types of warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

logging.basicConfig(filename='rnn_vae_fabric.log', level=logging.DEBUG)

# Get the current date and time
now = datetime.datetime.now()

def set_device(counters={"gpu_count": 0, "cpu_count": 0}):
  
    # make sure torch uses cuda for GPU computing
    use_gpu = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and not use_gpu

    if use_gpu:
        device = torch.device("cuda")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        counters["gpu_count"] += 1
        #if counters["gpu_count"] == 1:
            #print("Using CUDA")
            #print('GPU active:', torch.cuda.is_available())
            #print('GPU used:', torch.cuda.get_device_name(0))
    elif use_mps:
        device = torch.device("mps")
        torch.set_default_tensor_type('torch.FloatTensor')
        counters["gpu_count"] += 1
        #if counters["gpu_count"] == 1:
            #print("Using MPS")
    else:
        device = torch.device("cpu")
        counters["cpu_count"] += 1
        #if counters["cpu_count"] == 1:
        #    print("Using CPU")
        
    return device, use_gpu, use_mps

def to_tensor_if_not(tensor, device):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor([tensor], device=device)
    return tensor

def all_elements_are_tensors(tensor_list):
    return all(torch.is_tensor(x) for x in tensor_list)

""""
def save_losses_to_disk(loss_lists, model_name, cfg):
    """
    #Convert loss lists to tensors, move them to CPU, convert to NumPy arrays and save to disk.
    
    #Parameters:
    #    loss_lists (dict): Dictionary containing loss lists to be saved
    #    model_name (str): Name of the model
    #    cfg (dict): Configuration dictionary containing the project path
    
    #Returns:
    #   None
"""
    for loss_name, loss_list in loss_lists.items():

        # Convert list to tensor
        loss_tensor = torch.tensor(loss_list)
        
        # Convert tensor to numpy array and move to CPU
        loss_array = loss_tensor.cpu().numpy()
        
        # Save the numpy array to disk
        save_path = os.path.join(cfg['project_path'], 'model', 'model_losses', f"{loss_name}_{model_name}.npy")
        np.save(save_path, loss_array)
    
"""
def save_losses_to_disk(loss_lists, model_name, cfg, device):
    for loss_name, loss_list in loss_lists.items():
       
        if not all_elements_are_tensors(loss_list):
            fabric.print(f"Loss list '{loss_name}' contains non-tensor elements")
            try:
                for i, loss in enumerate(loss_list):
                    loss_list[i] = to_tensor_if_not(loss, device)
            except Exception as e:
                fabric.print(f"Converting loss list '{loss_name}' to tensor failed. Error: {e}")
                continue
        
        # Convert list to tensor by stacking if they are not already a single tensor
        if len(loss_list) > 0:
            if isinstance(loss_list[0], torch.Tensor):
                loss_tensor = torch.stack(loss_list)
            else:
                loss_tensor = torch.tensor(loss_list)
        
            # Convert tensor to numpy array and move to CPU
            loss_array = loss_tensor.cpu().numpy()
           
            try:
                # Save the numpy array to disk
                save_path = os.path.join(cfg['project_path'], 'model', 'model_losses', f"{loss_name}_{model_name}.npy")
                np.save(save_path, loss_array)
            except Exception as e:
                fabric.print(f"Saving loss array '{loss_name}' failed. Error: {e}")


def manage_and_save_model(fabric, train_start, epoch, avg_weight, avg_test_mse_loss, model, cfg, convergence, conv_counter, model_name, BEST_LOSS, SNAPSHOT):
                """
                Manages model saving based on conditions and handles convergence counter.

                Parameters:
                    epoch (int): Current epoch number.
                    avg_weight (float): Average weight value.
                    avg_test_mse_loss (float): Average test MSE loss.
                    model (PyTorch model): The model to be saved.
                    cfg (dict): Configuration dictionary.
                    convergence (int): Current convergence counter.
                    conv_counter (list): List to store convergence counters.
                    BEST_LOSS (float): Best loss achieved so far.
                    SNAPSHOT (int): Epoch interval for saving model snapshots.

                Returns:
                    int: Updated convergence counter.
                    float: Updated BEST_LOSS value.
                """
                 # Check conditions for best loss
                if avg_weight.item() > 0.99 and avg_test_mse_loss.item() <= BEST_LOSS:
                    BEST_LOSS = avg_test_mse_loss
                    fabric.print("Saving model!")
                    save_path = os.path.join(cfg['project_path'], "model", "best_model", model_name + '_' + cfg['Project'] + '_epoch_' + str(epoch) + '_time_' + str(train_start) + '.pkl')
                    try:
                        fabric.save(path=save_path, state=model.state_dict())
                    except Exception as e:
                        fabric.print(f"Saving model failed. Error: {e}")
                    convergence = 0
                else:
                    convergence += 1

                conv_counter.append(convergence)

                # Distribute the convergence across all processes
                convergence = fabric.broadcast(convergence, src=0)

                try: 
                    # Save model snapshot
                    if epoch % SNAPSHOT == 0:
                        fabric.print("Saving model snapshot!")
                        snapshot_path = os.path.join(cfg['project_path'], 'model', 'best_model', 'snapshots', model_name + '_' + cfg['Project'] + '_epoch_' + str(epoch) + '_time_' + str(train_start) + '.pkl')
                        try:
                            fabric.save(path=snapshot_path, state=model.state_dict())
                        except Exception as e:
                            fabric.print(f"Saving model snapshot failed. Error: {e}")
                except Exception as e:
                    fabric.print(f"Saving model snapshot failed. Error: {e}")

                return convergence, BEST_LOSS, conv_counter

def check_convergence(fabric, convergence, cfg):
    """
    Checks if the model has converged.
    
    Parameters:
        convergence (int): The current convergence counter value.
        cfg (dict): The configuration dictionary containing model settings.
        
    Returns:
        bool: True if the model has converged, False otherwise.
    """
    model_convergence_threshold = cfg['model_convergence']
    try:
        if convergence > model_convergence_threshold:
            fabric.print('Finished training...')
            fabric.print('Model converged. Please check your model with vame.evaluate_model(). \n'
                'You can also re-run vame.trainmodel() to further improve your model. \n'
                'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                'and plug your current model name into _pretrained_model_. \n'
                'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                '\n'
                'Next: \n'
                'Use vame.pose_segmentation() to identify behavioral motifs in your dataset!')
            
            return True
    except Exception as e:
        fabric.print(f"Checking convergence failed. Error: {e}")
        return False
        
    return False


def reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss

def future_reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss

def cluster_loss(H, kloss, lmbda, batch_size):
    gram_matrix = (H.T @ H) / batch_size
    _ ,sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:kloss])
    loss = torch.sum(sv)
    return lmbda*loss


def kullback_leibler_loss(mu, logvar):
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def kl_annealing(epoch, kl_start, annealtime, function):
    """
        Annealing of Kullback-Leibler loss to let the model learn first
        the reconstruction of the data before the KL loss term gets introduced.
    """
    if epoch > kl_start:
        if function == 'linear':
            new_weight = min(1, (epoch-kl_start)/(annealtime))

        elif function == 'sigmoid':
            new_weight = float(1/(1+np.exp(-0.9*(epoch-annealtime))))
        else:
            raise NotImplementedError('currently only "linear" and "sigmoid" are implemented')

        return new_weight

    else:
        new_weight = 0
        return new_weight


def gaussian(ins, is_training, seq_len, std_n=0.8):
    if is_training:
        emp_std = ins.std(1)*std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len)
        emp_std = emp_std.permute(0,2,1)
        noise = Variable(ins.data.new(ins.size()).normal_(0, 1))
        return ins + (noise*emp_std)
    return ins


def train(fabric, train_loader, epoch, model, optimizer, anneal_function, BETA, kl_start,
          annealtime, seq_len, future_decoder, future_steps, scheduler, mse_red,
          mse_pred, kloss, klmbda, bsize, noise):
    """
    The `train` function is responsible for training a model using a given data loader and optimizing the model's parameters using an optimizer. 
    It calculates various loss values, including reconstruction loss, Kullback-Leibler loss, and cluster loss. The function also handles annealing 
    of the Kullback-Leibler loss to gradually introduce it during training. The training progress is logged using the WandB library.

    Inputs:
    - train_loader: The data loader object that provides the training data.
    - epoch: The current epoch number.
    - model: The model to be trained.
    - optimizer: The optimizer used to update the model's parameters.
    - anneal_function: The annealing function used to adjust the weight of the Kullback-Leibler loss.
    - BETA: The weight factor for the Kullback-Leibler loss.
    - kl_start: The epoch at which the Kullback-Leibler loss starts to be introduced.
    - annealtime: The duration over which the Kullback-Leibler loss weight is annealed.
    - seq_len: The length of the input sequence.
    - future_decoder: A boolean indicating whether the model has a future decoder.
    - future_steps: The number of future steps to predict.
    - scheduler: The learning rate scheduler.
    - mse_red: The reduction method for the reconstruction loss.
    - mse_pred: The reduction method for the future reconstruction loss.
    - kloss: The number of singular values to consider for the cluster loss.
    - klmbda: The weight factor for the cluster loss.
    - bsize: The batch size.
    - noise: A boolean indicating whether to add noise to the input data.

    Outputs:
    - kl_weight: The weight of the Kullback-Leibler loss.
    - train_loss: The average training loss.
    - klmbda * kmeans_losses / idx: The average cluster loss.
    - kullback_loss / idx: The average Kullback-Leibler loss.
    - mse_loss / idx: The average reconstruction loss.
    - fut_loss / idx: The average future reconstruction loss.
    """
    
    fabric.barrier()
    
    #print(f'In the training loop.. Epoch: {epoch}, global rank: {fabric.global_rank}')
    
    # toggle model to train mode
    model.train() 
    
    # Initialize loss variables
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    loss = 0.0
    seq_len_half = seq_len // 2
   
    # Set device and data type (dytype)
    device, use_gpu, use_mps = set_device()
    dtype = None
    dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

    for idx, data_item in enumerate(train_loader):
        # Move data to device
        data_item = fabric.to_device(data_item) 
        data_item = Variable(data_item)
        data_item = data_item.permute(0,2,1)
        data = data_item[:,:seq_len_half,:].type(dtype)
        fut = data_item[:,seq_len_half:seq_len_half+future_steps,:].type(dtype)

        # Add noise to data if specified
        if noise:
            data_gaussian = gaussian(data,True,seq_len_half)
        else:
            data_gaussian = data

        # Determine whether to accumulate gradients
        is_accumulating = idx % 8 != 0  # Change '8' to whatever number of batches you want to accumulate
        
        # is_accumulating is used to determine if you're in the middle of a gradient accumulation process. 
        # If True, no_backward_sync will be enabled, reducing inter-process communication.
        
        # Wrap the forward and backward passes
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            
            if future_decoder:
                data_tilde, future, latent, mu, logvar = model(data_gaussian)
                rec_loss = reconstruction_loss(data, data_tilde, mse_red)
                fut_rec_loss = future_reconstruction_loss(fut, future, mse_pred)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
                loss = rec_loss + fut_rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
                fut_loss += fut_rec_loss.detach()
            else:
                data_tilde, latent, mu, logvar = model(data_gaussian)
                rec_loss = reconstruction_loss(data, data_tilde, mse_red) 
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
                loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss

            optimizer.zero_grad()
            fabric.backward(loss)
            
        # Step the optimizer every 8 batches (or whatever your condition is)
        if not is_accumulating:
            # Gradient clipping
            fabric.clip_gradients(model, optimizer, max_norm=5.0, norm_type='inf')

            # Update model parameters
            optimizer.step()
        
       
        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss.item()

    train_log_dict = {
        'train_loss': train_loss / idx,
        'train_mse_loss': mse_loss / idx,
        'train_kl_loss': BETA * kl_weight * kullback_loss / idx,
        'train_kmeans_loss': kl_weight * kmeans_losses / idx,
        'train_fut_loss': fut_loss / idx
    }

    wandb.log(train_log_dict)

        # if idx % 1000 == 0:
        #     print('Epoch: %d.  loss: %.4f' %(epoch, loss.item()))
   
    scheduler.step(loss) #be sure scheduler is called before optimizer in >1.1 pytorch
    
    
    return kl_weight, train_loss / idx, kl_weight * kmeans_losses / idx, kullback_loss / idx, mse_loss / idx, fut_loss / idx



def test(fabric, test_loader, epoch, model, optimizer, BETA, kl_weight, seq_len, mse_red, kloss, klmbda, future_decoder, bsize):
    model.eval() # toggle model to inference mode
    test_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)
    
    #print(f'In the testing loop.. Epoch: {epoch}, global rank: {fabric.global_rank}')
    with torch.no_grad():
        # we're only going to infer, so no autograd at all required
        # make sure torch uses cuda or MPS for GPU computing
        device, use_gpu, use_mps = set_device()
        dtype = None
        dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        
        for idx, data_item in enumerate(test_loader):
            data_item = fabric.to_device(data_item) 
            data_item = Variable(data_item)
            data_item = data_item.permute(0,2,1)
            data = data_item[:,:seq_len_half,:].type(dtype)

            if future_decoder:
                recon_images, _, latent, mu, logvar = model(data)
            else:
                recon_images, latent, mu, logvar = model(data)

            rec_loss = reconstruction_loss(data, recon_images, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            loss = rec_loss + BETA * kl_weight * kl_loss + kl_weight * kmeans_loss
            

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)

            test_loss += loss.item()
            mse_loss += rec_loss.item()
            kullback_loss += kl_loss.item()
            kmeans_losses += kmeans_loss
        
        test_log_dict = {
            'test_loss': test_loss / idx,
            'test_mse_loss': mse_loss / idx,
            'test_kl_loss': BETA * kl_weight * kullback_loss / idx,
            'test_kmeans_loss': kl_weight * kmeans_losses / idx
            }
        
        #fabric.log_dict(test_log_dict)
        wandb.log(test_log_dict)

    
    
    return mse_loss / idx, test_loss / idx, kl_weight * kmeans_losses / idx


def train_model(config):
    
    num_workers = 4
    train_start = time.time()
    
    fabric.print("Fabric initialized with %d processes" % fabric.world_size)
    fabric.print("Train model called...")
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    legacy = cfg['legacy']
    model_name = cfg['model_name']
    pretrained_weights = cfg['pretrained_weights']
    pretrained_model = cfg['pretrained_model']
    fixed = cfg['egocentric_data']

    """ W&B """
    # Initialize wandb logging on each node
    fabric.print("Logging in to wandb...")
    wandb.login(key=cfg['wandb_api_key'])

    fabric.print("Initializing wandb run...")

    try:
        wandb.init(
            project=cfg['wandb_project'],
            name=model_name + '__local_rank__' + str(fabric.local_rank) + '__global_rank__' + str(fabric.global_rank) + '_date_' + time.strftime('%Y-%m-%d_%H-%M-%S'),
            entity=cfg['wandb_entity'],
            group='DDP_1',
            config=cfg,
            reinit=True
        )
    except Exception as e:
        fabric.print("Wandb init failed. Check your API key and project name.")
        logging.debug(f"Wandb init failed. Error: {e}")
        return

    fabric.print("Train Variational Autoencoder - model name: %s \n" %model_name)
    os.makedirs(os.path.join(cfg['project_path'],'model','best_model'), exist_ok=True)
    os.makedirs(os.path.join(cfg['project_path'],'model','best_model','snapshots'), exist_ok=True)
    os.makedirs(os.path.join(cfg['project_path'],'model','model_losses',""), exist_ok=True)

    #device, use_gpu, use_mps = set_device()

    """ HYPERPARAMTERS """
    # General
    # CUDA = use_gpu
    SEED = 19
    TRAIN_BATCH_SIZE = cfg['batch_size']
    TEST_BATCH_SIZE = int(cfg['batch_size']/4)
    EPOCHS = cfg['max_epochs']
    ZDIMS = cfg['zdims']
    BETA  = cfg['beta']
    SNAPSHOT = cfg['model_snapshot']
    LEARNING_RATE = cfg['learning_rate']
    NUM_FEATURES = cfg['num_features']
    if fixed == False:
        NUM_FEATURES = NUM_FEATURES - 2
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_DECODER = cfg['prediction_decoder']
    FUTURE_STEPS = cfg['prediction_steps']
    
    # RNN
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    noise = cfg['noise']
    scheduler_step_size = cfg['scheduler_step_size']
    scheduler_thresh = cfg['scheduler_threshold']
    softplus = cfg['softplus']

    # Loss
    MSE_REC_REDUCTION = cfg['mse_reconstruction_reduction']
    MSE_PRED_REDUCTION = cfg['mse_prediction_reduction']
    KMEANS_LOSS = cfg['kmeans_loss']
    KMEANS_LAMBDA = cfg['kmeans_lambda']
    KL_START = cfg['kl_start']
    ANNEALTIME = cfg['annealtime']
    anneal_function = cfg['anneal_function']
    optimizer_scheduler = cfg['scheduler']

    BEST_LOSS = 999999
    convergence = 0
    fabric.print('Latent Dimensions: %d, Time window: %d, Batch Size: %d, Beta: %d, lr: %.4f\n' %(ZDIMS, cfg['time_window'], TRAIN_BATCH_SIZE, BETA, LEARNING_RATE))
    
    # simple logging of diverse losses.
    avg_train_losses = []
    avg_train_kmeans_losses = []
    avg_train_kl_losses = []
    avg_train_mse_losses = []
    avg_train_fut_losses = []
    avg_weight_values = []

    avg_test_losses = []
    avg_test_mse_losses = []
    avg_test_km_losses = []

    learn_rates = []
    conv_counter = []

    """ SEED """
    fabric.seed_everything(SEED)

    """ Model and Optimizer """
    RNN = RNN_VAE if not legacy else RNN_VAE_LEGACY
    
    model = RNN(TEMPORAL_WINDOW, ZDIMS, NUM_FEATURES, FUTURE_DECODER, FUTURE_STEPS, hidden_size_layer_1,
                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                dropout_rec, dropout_pred, softplus)


    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    fabric.print("Moving model and optimizer to device...")  

    model, optimizer = fabric.setup(model, optimizer, move_to_device=True)

    """ LOAD PRETRAINED WEIGHTS """
    if pretrained_weights:
        try:
            fabric.print("Loading pretrained weights from model: %s\n" %os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl'))
            model.load_state_dict(fabric.load(os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl')))
            KL_START = 0
            ANNEALTIME = 1
        except:
            fabric.print("No file found at %s\n" %os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl'))
            try:
                fabric.print("Loading pretrained weights from %s\n" %pretrained_model)
                model.load_state_dict(fabric.load(pretrained_model))
                KL_START = 0
                ANNEALTIME = 1
            except:
                fabric.print("Could not load pretrained model. Check file path in config.yaml.")
            
    """ DATASET """
    trainset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='train_seq.npy', train=True, temporal_window=TEMPORAL_WINDOW)
    testset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='test_seq.npy', train=False, temporal_window=TEMPORAL_WINDOW)

    #if device == torch.device("cuda"):
    cuda_generator = torch.Generator(device='cuda')
    train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers, generator=cuda_generator)
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers, generator=cuda_generator)
    #else:
        #cpu_generator = Generator(device='cpu')
        #train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers, generator=cpu_generator)
        #test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers, generator=cpu_generator)
    
    # Move dataloaders to device
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader, move_to_device=True, use_distributed_sampler=False)


    """ SCHEDULER """
    if optimizer_scheduler:
        fabric.print('Scheduler step size: %d, Scheduler gamma: %.2f, Scheduler Threshold: %.5f\n' %(scheduler_step_size, cfg['scheduler_gamma'], scheduler_thresh))
	# Thanks to @alexcwsmith for the optimized scheduler contribution
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=cfg['scheduler_gamma'], patience=cfg['scheduler_step_size'], threshold=scheduler_thresh, threshold_mode='rel', verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=1, last_epoch=-1)

    """ TRAINING """    

    wandb.watch(model, log='all')

    device = fabric.device
    
    fabric.print("Start training... ")

    fabric.barrier()
    
    for epoch in range(1, EPOCHS):
        
        fabric.print(f'Epoch: {epoch}, Epochs on convergence counter: {convergence}')
        
        
        #print(f'Training: {fabric.global_rank}')
        weight, train_loss, train_km_loss, kl_loss, mse_loss, fut_loss = train(fabric, train_loader, epoch, model, optimizer, 
                                                                            anneal_function, BETA, KL_START, 
                                                                            ANNEALTIME, TEMPORAL_WINDOW, FUTURE_DECODER,
                                                                            FUTURE_STEPS, scheduler, MSE_REC_REDUCTION,
                                                                            MSE_PRED_REDUCTION, KMEANS_LOSS, KMEANS_LAMBDA,
                                                                            TRAIN_BATCH_SIZE, noise)
        
        #print(f'End of training loop.. Epoch: {epoch}, global rank: {fabric.global_rank}')
        
    
        
        #print(f'Testing: {fabric.global_rank}')
        test_mse_loss, test_loss, test_km_loss = test(fabric, test_loader, epoch, model, optimizer,
                                                    BETA, weight, TEMPORAL_WINDOW, MSE_REC_REDUCTION,
                                                    KMEANS_LOSS, KMEANS_LAMBDA, FUTURE_DECODER, TEST_BATCH_SIZE)
        
        #print(f'End of testing loop.. Epoch: {epoch}, global rank: {fabric.global_rank}')

        # Convert to tensor if not already
        train_loss = to_tensor_if_not(train_loss, device)
        train_km_loss = to_tensor_if_not(train_km_loss, device)
        kl_loss = to_tensor_if_not(kl_loss, device)
        mse_loss = to_tensor_if_not(mse_loss, device)
        fut_loss = to_tensor_if_not(fut_loss, device)
        test_loss = to_tensor_if_not(test_loss, device)
        test_mse_loss = to_tensor_if_not(test_mse_loss, device)
        test_km_loss = to_tensor_if_not(test_km_loss, device)
        weight = to_tensor_if_not(weight, device)
           
        # Pause the processeses at this point so synchronization of the losses can be done across all processes
        # Average the losses across all processes
        fabric.barrier()
        avg_train_loss = fabric.all_reduce(train_loss, reduce_op='mean')
        fabric.barrier()
        avg_train_km_loss = fabric.all_reduce(train_km_loss, reduce_op='mean')
        fabric.barrier()
        avg_train_kl_loss = fabric.all_reduce(kl_loss, reduce_op='mean')
        fabric.barrier()
        avg_train_mse_loss = fabric.all_reduce(mse_loss, reduce_op='mean')
        fabric.barrier()
        avg_train_fut_loss = fabric.all_reduce(fut_loss, reduce_op='mean')
        fabric.barrier()
        avg_weight = fabric.all_reduce(weight, reduce_op='mean')
        fabric.barrier()
        avg_test_loss = fabric.all_reduce(test_loss, reduce_op='mean')
        fabric.barrier()
        avg_test_mse_loss = fabric.all_reduce(test_mse_loss, reduce_op='mean')
        fabric.barrier()
        avg_test_km_loss = fabric.all_reduce(test_km_loss, reduce_op='mean')
        fabric.barrier()

        fabric.print('Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(avg_train_loss.item(),
            avg_train_mse_loss.item(), avg_train_fut_loss.item(), avg_train_kl_loss.item(), avg_train_km_loss.item(), avg_weight.item()))
        
        fabric.print('Test loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}'.format(avg_test_loss.item(),
                    avg_test_mse_loss.item(), avg_test_km_loss.item(), avg_test_km_loss.item()))

        #print(f"Before barrier for epoch metrics, rank: {fabric.global_rank}")
        fabric.barrier()
        #print(f"After barrier for epoch metrics, rank: {fabric.global_rank}")
        epoch_metrics = {
                    "epoch": epoch,
                    "avg_train_loss": avg_train_loss,
                    "avg_train_mse_loss": avg_train_mse_loss,
                    "avg_train_fut_loss": avg_train_fut_loss,
                    "avg_train_kl_loss": avg_train_kl_loss,
                    "avg_train_kmeans_loss": avg_train_km_loss,
                    "avg_weight": avg_weight,
                    "avg_test_loss": avg_test_loss,
                    "avg_test_mse_loss": avg_test_mse_loss,
                    "avg_test_kl_loss": avg_test_km_loss,
                    "avg_test_kmeans_loss": avg_test_km_loss,
                    }
        

        wandb.log(epoch_metrics)

        avg_train_losses.append(avg_train_loss.item())
        avg_train_kmeans_losses.append(avg_train_km_loss.item())
        avg_train_kl_losses.append(avg_train_kl_loss.item())
        avg_weight_values.append(avg_weight.item())
        avg_train_mse_losses.append(avg_train_mse_loss.item())
        avg_train_fut_losses.append(avg_train_fut_loss.item())

        avg_test_losses.append(avg_test_loss.item())
        avg_test_mse_losses.append(avg_test_mse_loss.item())
        avg_test_km_losses.append(avg_test_km_loss.item())
        
        lr = optimizer.param_groups[0]['lr']
        learn_rates.append(lr)

        """ Saving the best model yet """
            # Check conditions for best loss
        if avg_weight.item() > 0.99 and avg_test_mse_loss.item() <= BEST_LOSS:
            BEST_LOSS = avg_test_mse_loss
            fabric.print("Saving model!")
            save_path = os.path.join(cfg['project_path'], "model", "best_model", model_name + '_' + cfg['Project'] + '_epoch_' + str(epoch) + '.pkl')
            fabric.save(path=save_path, state=model.state_dict())
            #fabric.print("Model saved!")    #fabric.log_dict(epoch_metrics)

            convergence = 0
        else:
            convergence += 1
        
        #print(f"Before barrier for convergence, rank: {fabric.global_rank}")
        fabric.barrier()
        #print(f"After barrier for convergence, rank: {fabric.global_rank}")
        conv_counter.append(convergence)

        """ Saving the model at the checkpoint """
        # Save model snapshot
        if epoch % SNAPSHOT == 0:
            fabric.print("Saving model snapshot!")
            snapshot_path = os.path.join(cfg['project_path'], 'model', 'best_model', 'snapshots', model_name + '_' + cfg['Project'] + '_epoch_' + str(epoch) + '.pkl')
            fabric.save(path=snapshot_path, state=model.state_dict())
            
        
        #print(f"Before barrier for snapshot saving, rank: {fabric.global_rank}")
        fabric.barrier()
        #print(f"After barrier for snapshot saving, rank: {fabric.global_rank}")

        loss_lists = {
            'avg_train_losses': avg_train_losses,
            'avg_test_losses': avg_test_losses,
            'avg_train_kmeans_losses': avg_train_kmeans_losses,
            'avg_train_kl_losses': avg_train_kl_losses,
            'avg_weight_values': avg_weight_values,
            'avg_train_mse_losses': avg_train_mse_losses,
            'avg_test_mse_losses': avg_test_mse_losses,
            'avg_train_fut_losses': avg_train_fut_losses,
            'avg_test_km_losses': avg_test_km_losses,
        }



        for loss_name, loss_list in loss_lists.items():

            if not all_elements_are_tensors(loss_list):
                #fabric.print(f"Loss list '{loss_name}' contains non-tensor elements")
                for i, loss in enumerate(loss_list):
                    loss_list[i] = to_tensor_if_not(loss, device)
            
            # Convert list to tensor by stacking if they are not already a single tensor
            if len(loss_list) > 0:
                if isinstance(loss_list[0], torch.Tensor):
                    loss_tensor = torch.stack(loss_list)
                else:
                    loss_tensor = torch.tensor(loss_list)
            
                # Convert tensor to numpy array and move to CPU
                loss_array = loss_tensor.cpu().numpy()
            
                # Save the numpy array to disk
                save_path = os.path.join(cfg['project_path'], 'model', 'model_losses', f"{loss_name}_{model_name}.npy")
                if fabric.global_rank == 0:
                    #print("Saving losses to disk...")
                    np.save(save_path, loss_array)
                    #print("Losses saved!")
        
        
        
        #print(f"Before barrier for losses list, rank: {fabric.global_rank}")
        fabric.barrier()
        #print(f"After barrier for losses list, rank: {fabric.global_rank}")
                

        
        
        
        if fabric.global_rank == 0:
            df = pd.DataFrame([avg_train_losses, avg_test_losses, avg_train_kmeans_losses, avg_train_kl_losses, avg_weight_values, avg_train_mse_losses, avg_train_fut_losses, learn_rates, conv_counter]).T
            df.columns=['Train_losses', 'Test_losses', 'Kmeans_losses', 'KL_losses', 'Weight_values', 'MSE_losses', 'Future_losses', 'Learning_Rate', 'Convergence_counter']
            print("Saving dataframe to disk...")
            try:
                df.to_csv(cfg['project_path']+'/model/model_losses/'+model_name+'_LossesSummary.csv')     
            except Exception as e:
                fabric.print("Could not save dataframe to disk. Error: %s" %e)
                logging.debug(f"Could not save dataframe to disk. Error: {e}")
                print("Data frame saved!")
        
        #print(f"Before barrier for dataframe saving, rank: {fabric.global_rank}")
        fabric.barrier()
        #print(f"After barrier for dataframe saving, rank: {fabric.global_rank}")
        
        #print(f"Process {fabric.global_rank} before convergence check")

        if convergence > cfg['model_convergence']:
            fabric.print('Finished training...')
            fabric.print('Model converged. Please check your model with vame.evaluate_model(). \n'
                'You can also re-run vame.trainmodel() to further improve your model. \n'
                'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                'and plug your current model name into _pretrained_model_. \n'
                'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                '\n'
                'Next: \n'
                'Use vame.pose_segmentation() to identify behavioral motifs in your dataset!')
            break
                
        #print(f"Process {fabric.global_rank} after convergence check") 

        fabric.print("Convengence check complete.")
            
            # Distribute the convergence across all processes
        convergence = fabric.broadcast(convergence, src=0)
        fabric.print("\n")
        
        #print(f"Before barrier for convergence check, rank: {fabric.global_rank}")
        fabric.barrier()
        #print(f"After barrier for convergence check, rank: {fabric.global_rank}")
        
        # Pause all processes that are not the main process until the main process reaches this point
        # This is done to ensure that the main process has finished logging the losses to wandb and fabric
        # before the other processes continue with the next epoch
        

        print(f"Continuing to next epoch... {epoch+1}, global rank: {fabric.global_rank}")
        
    fabric.barrier()
        
    if convergence < cfg['model_convergence']:
        fabric.print('Model seemed to have not reached convergence. You may want to check your model \n'
            'with vame.evaluate_model(). If your satisfied you can continue with \n'
            'Use vame.behavior_segmentation() to identify behavioral motifs!\n\n'
            'OPTIONAL: You can re-run vame.rnn_model() to improve performance.')

    wandb.finish()

if __name__ == "__main__":
    #config = "/Volumes/G-DRIVE_SSD/VAME_working/ALR_VAME_1-Sep15-2023/config_fabric.yaml"
    config= "/work/wachslab/aredwine3/VAME_working/config_fabric_2.yaml"

    SLURMEnvironment(train_model(config))

""" Lighting Fabric Options
  --accelerator [cpu|gpu|cuda|mps|tpu]
                                  The hardware accelerator to run on.

  --strategy [ddp|dp|deepspeed|deepspeed_stage_1|deepspeed_stage_2|deepspeed_stage_3|fsdp]
                                  Strategy for how to run across multiple
                                  devices.

  --devices TEXT                  Number of devices to run on (``int``), which
                                  devices to run on (``list`` or ``str``), or
                                  ``'auto'``. The value applies per node.

  --num-nodes, --num_nodes INTEGER
                                  Number of machines (nodes) for distributed
                                  execution.

  --node-rank, --node_rank INTEGER
                                  The index of the machine (node) this command
                                  gets started on. Must be a number in the
                                  range 0, ..., num_nodes - 1.

  --main-address, --main_address TEXT
                                  The hostname or IP address of the main
                                  machine (usually the one with node_rank =
                                  0).

  --main-port, --main_port INTEGER
                                  The main port to connect to the main
                                  machine.

  --precision [16-mixed|bf16-mixed|32-true|64-true|64|32|16|bf16]
                                  Double precision (``64-true`` or ``64``),
                                  full precision (``32-true`` or ``64``), half
                                  precision (``16-mixed`` or ``16``) or
                                  bfloat16 precision (``bf16-mixed`` or
                                  ``bf16``)
"""
