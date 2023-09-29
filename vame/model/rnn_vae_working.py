#!/usr/bin/env python3
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
import wandb
import torch
import yaml
import getpass
import numpy as np
import pandas as pd
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# Local application/library specific imports
from vame.util.auxiliary import read_config
from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE, RNN_VAE_LEGACY

# Warnings
import warnings

# Ignore these specific types of warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


def set_device(counters={"gpu_count": 0, "cpu_count": 0}):
  
    # make sure torch uses cuda for GPU computing
    use_gpu = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and not use_gpu

    if use_gpu:
        device = torch.device("cuda")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        counters["gpu_count"] += 1
        if counters["gpu_count"] == 1:
            print("Using CUDA")
            print('GPU active:', torch.cuda.is_available())
            print('GPU used:', torch.cuda.get_device_name(0))
    elif use_mps:
        device = torch.device("mps")
        torch.set_default_tensor_type('torch.FloatTensor')
        counters["gpu_count"] += 1
        if counters["gpu_count"] == 1:
            print("Using MPS")
    else:
        device = torch.device("cpu")
        counters["cpu_count"] += 1
        if counters["cpu_count"] == 1:
            print("Using CPU")
        
    return device, use_gpu, use_mps


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

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(train_loader, epoch, model, optimizer, anneal_function, BETA, kl_start,
          annealtime, seq_len, future_decoder, future_steps, scheduler, mse_red, 
          mse_pred, kloss, klmbda, bsize, noise):
    model.train() # toggle model to train mode
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    # fut_loss = torch.zeros(1).to(device) #TRY NEXT
    loss = 0.0
    seq_len_half = int(seq_len / 2)

    device, use_gpu, use_mps = set_device()
    # make sure torch uses cuda or MPS for GPU computing
    #use_gpu = torch.cuda.is_available()
    #use_mps = torch.backends.mps.is_available() and not use_gpu

    dtype = None
    if use_gpu:
        dtype = torch.cuda.FloatTensor
    elif use_mps:
        dtype = torch.FloatTensor
    else:
        dtype = torch.FloatTensor

    for idx, data_item in enumerate(train_loader):
        data_item = data_item.to(device)
        data_item = Variable(data_item)
        data_item = data_item.permute(0,2,1)
        data = data_item[:,:seq_len_half,:].type(dtype)
        fut = data_item[:,seq_len_half:seq_len_half+future_steps,:].type(dtype)


        if noise:
            data_gaussian = gaussian(data,True,seq_len_half)
        else:
            data_gaussian = data

        if future_decoder:
            data_tilde, future, latent, mu, logvar = model(data_gaussian)

            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            fut_rec_loss = future_reconstruction_loss(fut, future, mse_pred)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + fut_rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
            fut_loss += fut_rec_loss.detach()#.item()

        else:
            data_tilde, latent, mu, logvar = model(data_gaussian)

            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)

        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss.item()

        # if idx % 1000 == 0:
        #     print('Epoch: %d.  loss: %.4f' %(epoch, loss.item()))

        wandb.log({'batch_train_loss': loss.item()})
        wandb.log({'batch_train_mse_loss': rec_loss.item()})

   
    #scheduler.step(loss) #be sure scheduler is called before optimizer in >1.1 pytorch

    if future_decoder:
        print(time.strftime('%H:%M:%S'))
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
              mse_loss /idx, fut_loss/idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx, kl_weight))
    else:
        print(time.strftime('%H:%M:%S'))
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
              mse_loss /idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx, kl_weight))

    return kl_weight, train_loss/idx, kl_weight*kmeans_losses/idx, kullback_loss/idx, mse_loss/idx, fut_loss/idx


def test(test_loader, epoch, model, optimizer, BETA, kl_weight, seq_len, mse_red, kloss, klmbda, future_decoder, bsize):
    model.eval() # toggle model to inference mode
    test_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)
    device, use_gpu, use_mps = set_device()
    
    with torch.no_grad():
        # we're only going to infer, so no autograd at all required
        # make sure torch uses cuda or MPS for GPU computing
        #use_gpu = torch.cuda.is_available()
        #use_mps = torch.backends.mps.is_available() and not use_gpu

        dtype = None
        if use_gpu:
            dtype = torch.cuda.FloatTensor
        elif use_mps:
            dtype = torch.FloatTensor
        else:
            dtype = torch.FloatTensor

        for idx, data_item in enumerate(test_loader):
            data_item = data_item.to(device)
            data_item = Variable(data_item)
            data_item = data_item.permute(0,2,1)
            data = data_item[:,:seq_len_half,:].type(dtype)

            if future_decoder:
                recon_images, _, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, recon_images, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                loss = rec_loss + BETA*kl_weight*kl_loss+ kl_weight*kmeans_loss

            else:
                recon_images, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, recon_images, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)

            test_loss += loss.item()
            mse_loss += rec_loss.item()
            kullback_loss += kl_loss.item()
            kmeans_losses += kmeans_loss

            wandb.log({'batch_test_loss': loss.item()})
            wandb.log({'batch_test_mse_loss': rec_loss.item()})

    print('Test loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}'.format(test_loss / idx,
          mse_loss /idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx))

    return mse_loss /idx, test_loss/idx, kl_weight*kmeans_losses/idx


def train_model(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    model_name = cfg['model_name']
    pretrained_weights = cfg['pretrained_weights']
    pretrained_model = cfg['pretrained_model']
    fixed = cfg['egocentric_data']
    
    print("Train Variational Autoencoder - model name: %s \n" %model_name)
    os.makedirs(os.path.join(cfg['project_path'],'model','best_model'), exist_ok=True)
    os.makedirs(os.path.join(cfg['project_path'],'model','best_model','snapshots'), exist_ok=True)
    os.makedirs(os.path.join(cfg['project_path'],'model','model_losses',""), exist_ok=True)

    # make sure torch uses cuda or MPS for GPU computing
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
        print("warning, a GPU was not found... proceeding with CPU (slow!) \n")
        #raise NotImplementedError('GPU Computing is required!')
        
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
    print('Latent Dimensions: %d, Time window: %d, Batch Size: %d, Beta: %d, lr: %.4f\n' %(ZDIMS, cfg['time_window'], TRAIN_BATCH_SIZE, BETA, LEARNING_RATE))
    
    # simple logging of diverse losses
    train_losses = []
    test_losses = []
    kmeans_losses = []
    kl_losses = []
    weight_values = []
    mse_losses = []
    fut_losses = []
    learn_rates = []
    test_mse_losses = []
    conv_counter = []

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if legacy == False:
        RNN = RNN_VAE
    else:
        RNN = RNN_VAE_LEGACY
       
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(SEED)
    else:
        torch.manual_seed(SEED)
    
    # Initialize the model
    model = RNN(TEMPORAL_WINDOW, ZDIMS, NUM_FEATURES, FUTURE_DECODER, FUTURE_STEPS, hidden_size_layer_1,
                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                dropout_rec, dropout_pred, softplus)

    # Move the model to the appropriate device
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    if optimizer_scheduler:
        print('Scheduler step size: %d, Scheduler gamma: %.2f, Scheduler Threshold: %.5f\n' %(scheduler_step_size, cfg['scheduler_gamma'], scheduler_thresh))
	# Thanks to @alexcwsmith for the optimized scheduler contribution
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=cfg['scheduler_gamma'], patience=cfg['scheduler_step_size'], threshold=scheduler_thresh, threshold_mode='rel', verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=1, last_epoch=-1)

    if pretrained_weights:
        try:
            print("Loading pretrained weights from model: %s\n" %os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl'))
            model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl')))
            KL_START = 0
            ANNEALTIME = 1
        except:
            print("No file found at %s\n" %os.path.join(cfg['project_path'],'model','best_model',pretrained_model+'_'+cfg['Project']+'.pkl'))
            try:
                print("Loading pretrained weights from %s\n" %pretrained_model)
                model.load_state_dict(torch.load(pretrained_model))
                KL_START = 0
                ANNEALTIME = 1
            except:
                print("Could not load pretrained model. Check file path in config.yaml.")
            
    """ DATASET """
    trainset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='train_seq.npy', train=True, temporal_window=TEMPORAL_WINDOW)
    testset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='test_seq.npy', train=False, temporal_window=TEMPORAL_WINDOW)

    if device == torch.device("cuda"): 
        cuda_generator = torch.Generator(device='cuda')
        train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, generator=cuda_generator)
        test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, generator=cuda_generator)
    elif device == torch.device("mps"):
        mps_generator = torch.Generator(device='mps')
        train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, generator=mps_generator)
        test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, generator=mps_generator)
    else:
        train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    
    """ WANDB LOGGING """
    # Ask the user if the script should be run with wandb logging or hyperparameter optimization
    wandb_usage = input("Do you want to use wandb logging? (y/n) ").lower()

    if wandb_usage not in ['y', 'n']:
        raise ValueError("Please enter 'y', or 'n'.")

    if wandb_usage == 'l':
        wandb_project_name = input("Please enter your wandb project name: ")
        wandb_run_name = input("Please enter a name for your wandb run: ")
        wandb_username = input("Please enter your wandb username: ")
        wandb_api_key = getpass.getpass("Please enter your wandb API key: ")
        wandb.login(key=wandb_api_key)

        # Group all hyperparameters together in a dict

        config_dict = {
            'model_name': model_name,
            'pretrained_weights': pretrained_weights,
            'pretrained_model': pretrained_model,
            'SEED': SEED,
            'TRAIN_BATCH_SIZE': TRAIN_BATCH_SIZE,
            'TEST_BATCH_SIZE': TEST_BATCH_SIZE,
            'batch_size': cfg['batch_size'],
            'EPOCHS': EPOCHS,
            'ZDIMS': ZDIMS,
            'BETA': BETA,
            'SNAPSHOT': SNAPSHOT,
            'LEARNING_RATE': LEARNING_RATE,
            'NUM_FEATURES': NUM_FEATURES,
            'TEMPORAL_WINDOW': TEMPORAL_WINDOW,
            'FUTURE_DECODER': FUTURE_DECODER,
            'FUTURE_STEPS': FUTURE_STEPS,
            'hidden_size_layer_1': hidden_size_layer_1,
            'hidden_size_layer_2': hidden_size_layer_2,
            'hidden_size_rec': hidden_size_rec,
            'hidden_size_pred': hidden_size_pred,
            'dropout_encoder': dropout_encoder,
            'dropout_rec': dropout_rec,
            'dropout_pred': dropout_pred,
            'noise': noise,
            'sheduler_step_size': scheduler_step_size,
            'scheduler_thresh': scheduler_thresh,
            'softplus': softplus,
            'MSE_REC_REDUCTION': MSE_REC_REDUCTION,
            'MSE_PRED_REDUCTION': MSE_PRED_REDUCTION,
            'KMEANS_LOSS': KMEANS_LOSS,
            'KMEANS_LAMBDA': KMEANS_LAMBDA,
            'KL_START': KL_START,
            'ANNEALTIME': ANNEALTIME,
            'anneal_function': anneal_function,
            'optimizer_scheduler': optimizer_scheduler
        }
        
    # Print and confirm hyperparameters
        print("\nHere are the configurations:")
        for key, value in config_dict.items():
            print(f"{key}: {value}")
        
        confirm = input("\nDo these look correct? (y/n) ").lower()
        
        if confirm == 'y':
            wandb.init(project=wandb_project_name, entity=wandb_username, name=wandb_run_name)
            
            # Use dictionary unpacking to assign values to wandb.config
            wandb.config.update(config_dict)
        else:
            print("Please modify your configurations and try again.")
            exit()
            
    print("Start training... ")

    for epoch in range(1,EPOCHS):
        print('Epoch: %d' %epoch + ', Epochs on convergence counter: %d' %convergence)
        print('Train: ')
        weight, train_loss, km_loss, kl_loss, mse_loss, fut_loss = train(train_loader, epoch, model, optimizer, 
                                                                         anneal_function, BETA, KL_START, 
                                                                         ANNEALTIME, TEMPORAL_WINDOW, FUTURE_DECODER,
                                                                         FUTURE_STEPS, scheduler, MSE_REC_REDUCTION,
                                                                         MSE_PRED_REDUCTION, KMEANS_LOSS, KMEANS_LAMBDA,
                                                                         TRAIN_BATCH_SIZE, noise)

        test_mse_loss, test_loss, test_list = test(test_loader, epoch, model, optimizer,
                                                  BETA, weight, TEMPORAL_WINDOW, MSE_REC_REDUCTION,
                                                  KMEANS_LOSS, KMEANS_LAMBDA, FUTURE_DECODER, TEST_BATCH_SIZE)

        scheduler.step(test_loss) #be sure scheduler is called before optimizer in >1.1 pytorch
        
        # The scheduler.step() should be called after validating the model because 
        # we're using the ReduceLROnPlateau scheduler. This particular scheduler 
        # adjusts the learning rate based on a monitored metric (in this case, 
        # the validation (test) loss). Therefore, it's crucial to place scheduler.step(test_loss) 
        # after the validation loop has calculated the validation loss for the current epoch.
        
        # logging losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        kmeans_losses.append(km_loss)
        kl_losses.append(kl_loss)
        weight_values.append(weight)
        mse_losses.append(mse_loss)
        test_mse_losses.append(test_mse_loss)
        #fut_losses.append(fut_loss)
        fut_losses.append(fut_loss.cpu().item())

        lr = optimizer.param_groups[0]['lr']
        learn_rates.append(lr)

        if wandb_usage is 'l' and wandb.run is not None:
            wandb.log({'learning_rate': lr,
                    'train_loss': train_loss,
                    'mse_loss_train': mse_loss,
                    'test_loss': test_loss,
                    'mse_loss_test': test_mse_loss,
                    'kmeans_loss': km_loss,
                    'kl_loss': kl_loss,
                    'weight': weight,
                    'fut_loss': fut_loss,
                    'epoch': epoch,
                    'convergence': convergence
                    })
        # save best model
        if weight > 0.99 and test_mse_loss <= BEST_LOSS:
            BEST_LOSS = test_mse_loss
            print("Saving model!")
            torch.save(model.state_dict(), os.path.join(cfg['project_path'],"model", "best_model",model_name+'_'+cfg['Project']+'.pkl'))
            convergence = 0
        else:
            convergence += 1
        conv_counter.append(convergence)

        # save model snapshot
        if epoch % SNAPSHOT == 0:
            print("Saving model snapshot!\n")
            torch.save(model.state_dict(), os.path.join(cfg['project_path'],'model','best_model','snapshots',model_name+'_'+cfg['Project']+'_epoch_'+str(epoch)+'.pkl'))
            if wandb_usage:
                wandb.save(os.path.join(cfg['project_path'],'model','best_model','snapshots',model_name+'_'+cfg['Project']+'_epoch_'+str(epoch)+'.pkl'))

        # save logged losses
        np.save(os.path.join(cfg['project_path'],'model','model_losses','train_losses_'+model_name), train_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','test_losses_'+model_name), test_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','kmeans_losses_'+model_name), kmeans_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','kl_losses_'+model_name), kl_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','weight_values_'+model_name), weight_values)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','mse_train_losses_'+model_name), mse_losses)
        np.save(os.path.join(cfg['project_path'],'model','model_losses','mse_test_losses_'+model_name), test_mse_losses)
        # np.save(os.path.join(cfg['project_path'], 'model', 'model_losses', 'fut_losses_' + model_name), fut_losses)

        # Convert fut_losses to a tensor and save
        fut_losses_tensor = torch.tensor(fut_losses)
        fut_losses_array = fut_losses_tensor.cpu().detach().numpy()
        np.save(os.path.join(cfg['project_path'], 'model', 'model_losses', 'fut_losses_' + model_name), fut_losses_array)

        df = pd.DataFrame([train_losses, test_losses, kmeans_losses, kl_losses, weight_values, mse_losses, fut_losses, learn_rates, conv_counter]).T
        df.columns=['Train_losses', 'Test_losses', 'Kmeans_losses', 'KL_losses', 'Weight_values', 'MSE_losses', 'Future_losses', 'Learning_Rate', 'Convergence_counter']
        df.to_csv(cfg['project_path']+'/model/model_losses/'+model_name+'_LossesSummary.csv')     
        print("\n")

        if convergence > cfg['model_convergence']:
            print('Finished training...')
            print('Model converged. Please check your model with vame.evaluate_model(). \n'
                  'You can also re-run vame.trainmodel() to further improve your model. \n'
                  'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                  'and plug your current model name into _pretrained_model_. \n'
                  'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                  '\n'
                  'Next: \n'
                  'Use vame.pose_segmentation() to identify behavioral motifs in your dataset!')
            #return
            break

    if convergence < cfg['model_convergence']:
        print('Model seemed to have not reached convergence. You may want to check your model \n'
              'with vame.evaluate_model(). If your satisfied you can continue with \n'
              'Use vame.behavior_segmentation() to identify behavioral motifs!\n\n'
              'OPTIONAL: You can re-run vame.rnn_model() to improve performance.')
    
    if wandb_usage:    
        wandb.finish()
    


