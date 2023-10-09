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
from tempfile import TemporaryFile
import time
from pathlib import Path

# Third-party libraries
import wandb
import torch
import logging
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

# Generate a random number to use as a suffix for saving the model
random_suffix = str(np.random.randint(100))

# Set up logging
logging.basicConfig(
    filename=f'rnn_vae_wandb_sweeps_{random_suffix}.log', level=logging.DEBUG)


def set_device(counters={"gpu_count": 0, "cpu_count": 0}):

    # make sure torch uses cuda for GPU computing
    use_gpu = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and not use_gpu

    if use_gpu:
        device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # torch.set_default_device(f'cuda:{torch.cuda.current_device()}')
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)
        counters["gpu_count"] += 1
        if counters["gpu_count"] == 1:
            logging.info("Using CUDA")
            logging.info('GPU active: %s', torch.cuda.is_available())
            logging.info('GPU used: %s', torch.cuda.get_device_name(0))
    elif use_mps:
        device = torch.device("mps")
        # torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_default_device('mps')
        counters["gpu_count"] += 1
        if counters["gpu_count"] == 1:
            logging.info("Using MPS")
    else:
        device = torch.device("cpu")
        counters["cpu_count"] += 1
        if counters["cpu_count"] == 1:
            logging.info("Using CPU")

    return device, use_gpu, use_mps


def reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde, x)
    return rec_loss


def future_reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde, x)
    return rec_loss


def cluster_loss(H, kloss, lmbda, batch_size):
    gram_matrix = (H.T @ H) / batch_size
    _, sv_2, _ = torch.svd(gram_matrix)
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
            raise NotImplementedError(
                'currently only "linear" and "sigmoid" are implemented')

        return new_weight

    else:
        new_weight = 0
        return new_weight


def gaussian(ins, is_training, seq_len, std_n=0.8):
    if is_training:
        emp_std = ins.std(1)*std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len)
        emp_std = emp_std.permute(0, 2, 1)
        noise = Variable(ins.data.new(ins.size()).normal_(0, 1))
        return ins + (noise*emp_std)
    return ins


def train(train_loader, epoch, model, optimizer, anneal_function, BETA, kl_start,
          annealtime, seq_len, future_decoder, future_steps, scheduler, mse_red,
          mse_pred, kloss, klmbda, bsize, noise):
    logging.info(f"In the training loop. Epoch: {epoch}")

    model.train()  # toggle model to train mode
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)

    device, use_gpu, use_mps = set_device()

    dtype = None
    if use_gpu:
        dtype = torch.cuda.FloatTensor
    elif use_mps:
        dtype = torch.FloatTensor
    else:
        dtype = torch.FloatTensor

    for idx, data_item in enumerate(train_loader):
        data_item = Variable(data_item.to(device)).permute(0, 2, 1)
        data = data_item[:, :seq_len_half, :].type(dtype)
        fut = data_item[:, seq_len_half:seq_len_half +
                        future_steps, :].type(dtype)

        if noise:
            data_gaussian = gaussian(data, True, seq_len_half)
        else:
            data_gaussian = data

        if future_decoder:
            data_tilde, future, latent, mu, logvar = model(data_gaussian)

            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            fut_rec_loss = future_reconstruction_loss(fut, future, mse_pred)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kl_weight = kl_annealing(
                epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + fut_rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
            fut_loss += fut_rec_loss.detach()  # .item()

        else:
            data_tilde, latent, mu, logvar = model(data_gaussian)

            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_weight = kl_annealing(
                epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss.item()

        # if idx % 1000 == 0:
        #     logging.info('Epoch: %d.  loss: %.4f' %(epoch, loss.item()))

    # be sure scheduler is called before optimizer in >1.1 pytorch
    scheduler.step(loss)

    if future_decoder:
        logging.info(time.strftime('%H:%M:%S'))
        logging.info('Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
                                                                                                                                                 mse_loss / idx, fut_loss/idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx, kl_weight))
    else:
        logging.info(time.strftime('%H:%M:%S'))
        logging.info('Train loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(train_loss / idx,
                                                                                                                         mse_loss / idx, BETA*kl_weight*kullback_loss/idx, kl_weight*kmeans_losses/idx, kl_weight))

    return kl_weight, train_loss/idx, kl_weight*kmeans_losses/idx, kullback_loss/idx, mse_loss/idx, fut_loss/idx


def test(test_loader, epoch, model, optimizer, BETA, kl_weight, seq_len, mse_red, kloss, klmbda, future_decoder, bsize):
    model.eval()  # toggle model to inference mode
    test_loss = 0.0
    test_mse_loss = 0.0
    test_kullback_loss = 0.0
    test_kmeans_losses = 0.0
    loss = 0.0
    seq_len_half = int(seq_len / 2)

    device, use_gpu, use_mps = set_device()

    with torch.no_grad():
        # we're only going to infer, so no autograd at all required
        # make sure torch uses cuda or MPS for GPU computing
        use_gpu = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available() and not use_gpu

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
            data_item = data_item.permute(0, 2, 1)
            data = data_item[:, :seq_len_half, :].type(dtype)

            if future_decoder:
                recon_images, _, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, recon_images, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss

            else:
                recon_images, latent, mu, logvar = model(data)
                rec_loss = reconstruction_loss(data, recon_images, mse_red)
                kl_loss = kullback_leibler_loss(mu, logvar)
                kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
                loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)

            test_loss += loss.item()
            test_mse_loss += rec_loss.item()
            test_kullback_loss += kl_loss.item()
            test_kmeans_losses += kmeans_loss

    logging.info('Test loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}'.format(test_loss / idx,
                                                                                                    test_mse_loss / idx, BETA*kl_weight*test_kullback_loss/idx, kl_weight*test_kmeans_losses/idx))

    return test_mse_loss/idx, test_loss/idx, kl_weight*test_kmeans_losses/idx


sweep_configuration = {
    "method": "random",
    "metric": {
        "name": "train_loss",
        "goal": "minimize"
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3
    },
    "parameters": {
        "Project": {
            "values": ["ALR_VAME_1"],
            "distribution": "categorical"
        },
        "project_path": {
            "values": ["/work/wachslab/aredwine3/VAME_working/"],
            "distribution": "categorical"
        },
        "model_name": {
            "values": ["VAME_15prcnt_sweep"],
            "distribution": "categorical"
        },
        "pretrained_model": {
            "values": [False],
            "distribution": "categorical"
        },
        "pretrained_weights": {
            "values": [False],
            "distribution": "categorical"
        },
        "num_features": {
            "value": 28,
            "distribution": "constant"
        },
        "batch_size": {
            "values": [256, 512, 1024],
            "distribution": "categorical"
        },
        "max_epochs": {
            "value": 150,
            "distribution": "constant"
        },
        "model_snapshot": {
            "value": 15,
            "distribution": "constant"
        },
        "model_convergence": {
            "value": 25,
            "distribution": "constant"
        },
        "transition_function": {
            "values": ["GRU"],
            "distribution": "categorical"
        },
        "beta": {
            "value": 1,
            "distribution": "constant"
        },
        "beta_norm": {
            "values": [False],
            "distribution": "categorical"
        },
        "zdims": {
            "values": [10, 15, 20, 25, 30, 35, 40, 45, 50],
            "distribution": "categorical"
        },
        "learning_rate": {
            "values": [0.0001, 0.0005, 0.001, 0.005],
            "distribution": "categorical"
        },
        "time_window": {
            "values": [30, 45, 60, 75, 90, 105, 120],
            "distribution": "categorical"
        },
        "prediction_decoder": {
            "value": 1,

            "distribution": "constant"
        },
        "prediction_steps": {
            "values": [15, 20, 25, 30, 35],
            "distribution": "categorical"
        },
        "noise": {
            "values": [True, False],
            "distribution": "categorical"
        },
        "scheduler": {
            "value": 1,
            "distribution": "constant"
        },
        "scheduler_step_size": {
            "value": 100,
            "distribution": "constant"
        },
        "scheduler_gamma": {
            "value": 0.2,
            "distribution": "constant"
        },
        "scheduler_threshold": {
            "values": [0.0001, 0.001, 0.01, 0.05, 0.1],
            "distribution": "categorical"
        },
        "softplus": {
            "values": [True, False]
        },
        "hidden_layer_size_1": {
            "values": [256, 512],
            "distribution": "categorical"
        },
        "hidden_layer_size_2": {
            "values": [256, 512],
            "distribution": "categorical"
        },
        "dropout_encoder": {
            "values": [0, 1],
            "distribution": "categorical"
        },
        "hidden_size_rec": {
            "values": [256, 512],
            "distribution": "categorical"
        },
        "dropout_rec": {
            "values": [0, 1],
            "distribution": "categorical"
        },
        "n_layers": {
            "min": 2,
            "max": 4,
            "distribution": "int_uniform"
        },
        "hidden_size_pred": {
            "values": [256, 512],
            "distribution": "categorical"
        },
        "dropout_pred": {
            "values": [0, 1],
            "distribution": "categorical"
        },
        "mse_reconstruction_reduction": {
            "values": ["sum"],
            "distribution": "categorical"
        },
        "mse_prediction_reduction": {
            "values": ["sum"],
            "distribution": "categorical"
        },
        "kmeans_loss": {
            "value": 30,
            "distribution": "constant",
        },
        "kmeans_lambda": {
            "value": 0.1,
            "distribution": "constant"
        },
        "anneal_function": {
            "values": ["linear", "sigmoid"],
            "distribution": "categorical"
        },
        "kl_start": {
            "value": 2,
            "distribution": "constant"
        },
        "annealtime": {
            "value": 4,
            "distribution": "constant"
        },
        "legacy": {
            "values": [False],
            "distribution": "categorical"
        },
        "egocentric_data": {
            "values": [False],
            "distribution": "categorical"
        }
    }
}

wandb.login(key='bcd2a5a57142a0e6bb3d51242f679ab3d00dd8d4')

sweep_id = wandb.sweep(sweep=sweep_configuration,
                       project="VAME", entity="aredwine3")


def train_model():
    wandb.init(
        group="VAME_15prcnt_sweep",
    )

    legacy = wandb.config.legacy
    project = wandb.config.Project
    project_path = wandb.config.project_path
    model_name = wandb.config.model_name
    pretrained_weights = wandb.config.pretrained_weights
    pretrained_model = wandb.config.pretrained_model
    fixed = wandb.config.egocentric_data

    logging.info(
        "Train Variational Autoencoder - model name: %s \n" % model_name)
    os.makedirs(os.path.join(project_path, 'model',
                'best_model'), exist_ok=True)
    os.makedirs(os.path.join(project_path, 'model',
                'best_model', 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(project_path, 'model',
                'model_losses', ""), exist_ok=True)

    TRAIN_BATCH_SIZE = wandb.config.batch_size
    TEST_BATCH_SIZE = int(wandb.config.batch_size/4)
    EPOCHS = wandb.config.max_epochs
    ZDIMS = wandb.config.zdims
    BETA = wandb.config.beta
    SNAPSHOT = wandb.config.model_snapshot
    LEARNING_RATE = wandb.config.learning_rate
    NUM_FEATURES = wandb.config.num_features
    if fixed == False:
        NUM_FEATURES = NUM_FEATURES - 2
    TEMPORAL_WINDOW = wandb.config.time_window*2
    FUTURE_DECODER = wandb.config.prediction_decoder
    FUTURE_STEPS = wandb.config.prediction_steps
    model_convergence = wandb.config.model_convergence

    # RNN
    hidden_size_layer_1 = wandb.config.hidden_layer_size_1
    hidden_size_layer_2 = wandb.config.hidden_layer_size_2
    hidden_size_rec = wandb.config.hidden_size_rec
    hidden_size_pred = wandb.config.hidden_size_pred
    dropout_encoder = wandb.config.dropout_encoder
    dropout_rec = wandb.config.dropout_rec
    dropout_pred = wandb.config.dropout_pred
    noise = wandb.config.noise
    scheduler_gamma = wandb.config.scheduler_gamma
    scheduler_step_size = wandb.config.scheduler_step_size
    scheduler_thresh = wandb.config.scheduler_threshold
    softplus = wandb.config.softplus

    # Loss
    MSE_REC_REDUCTION = wandb.config.mse_reconstruction_reduction
    MSE_PRED_REDUCTION = wandb.config.mse_prediction_reduction
    KMEANS_LOSS = wandb.config.kmeans_loss
    KMEANS_LAMBDA = wandb.config.kmeans_lambda
    KL_START = wandb.config.kl_start
    ANNEALTIME = wandb.config.annealtime
    anneal_function = wandb.config.anneal_function
    optimizer_scheduler = wandb.config.scheduler

    device, use_gpu, use_mps = set_device()

    SEED = 19
    # CUDA = use_gpu

    BEST_LOSS = 999999
    convergence = 0
    logging.info('Latent Dimensions: %d, Time window: %d, Batch Size: %d, Beta: %d, lr: %.4f\n' % (
        ZDIMS, int(TEMPORAL_WINDOW/2), TRAIN_BATCH_SIZE, BETA, LEARNING_RATE))

    # simple logging of diverse losses
    train_losses = []
    train_kmeans_losses = []
    train_kl_losses = []
    weight_values = []
    train_mse_losses = []
    fut_losses = []
    learn_rates = []
    conv_counter = []

    test_losses = []
    test_mse_losses = []
    test_kmeans_losses = []

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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    if optimizer_scheduler:
        logging.info('Scheduler step size: %d, Scheduler gamma: %.2f, Scheduler Threshold: %.5f\n' % (
            scheduler_step_size, scheduler_gamma, scheduler_thresh))
    # Thanks to @alexcwsmith for the optimized scheduler contribution
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=scheduler_gamma, patience=scheduler_step_size,
                                      threshold=scheduler_thresh, threshold_mode='rel', verbose=True)
    else:
        scheduler = StepLR(
            optimizer, step_size=scheduler_step_size, gamma=1, last_epoch=-1)

    if pretrained_weights:
        try:
            logging.info("Loading pretrained weights from model: %s\n" % os.path.join(
                project_path, 'model', 'best_model', pretrained_model+'_'+project+'.pkl'))
            model.load_state_dict(torch.load(os.path.join(
                project_path, 'model', 'best_model', pretrained_model+'_'+project+'.pkl')))
            KL_START = 0
            ANNEALTIME = 1
        except:
            logging.info("No file found at %s\n" % os.path.join(
                project_path, 'model', 'best_model', pretrained_model+'_'+project+'.pkl'))
            try:
                logging.info("Loading pretrained weights from %s\n" %
                             pretrained_model)
                model.load_state_dict(torch.load(pretrained_model))
                KL_START = 0
                ANNEALTIME = 1
            except:
                logging.info(
                    "Could not load pretrained model. Check file path in config.yaml.")

    """ DATASET """
    trainset = SEQUENCE_DATASET(os.path.join(project_path, "data", "train", ""),
                                data='train_seq.npy', train=True, temporal_window=TEMPORAL_WINDOW)
    testset = SEQUENCE_DATASET(os.path.join(project_path, "data", "train", ""),
                               data='test_seq.npy', train=False, temporal_window=TEMPORAL_WINDOW)

    if device == torch.device("cuda"):
        cuda_generator = torch.Generator(device='cuda')
        train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                       shuffle=True, drop_last=True, num_workers=24, generator=cuda_generator)
        test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                      shuffle=True, drop_last=True, num_workers=24, generator=cuda_generator)
    elif device == torch.device("mps"):
        mps_generator = torch.Generator(device='mps')
        train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                       shuffle=True, drop_last=True, num_workers=4, generator=mps_generator)
        test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE,
                                      shuffle=True, drop_last=True, num_workers=4, generator=mps_generator)
    else:
        train_loader = Data.DataLoader(
            trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        test_loader = Data.DataLoader(
            testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    wandb.watch(model, log="all")

    logging.info("Start training... ")

    for epoch in range(1, EPOCHS):
        logging.info('Epoch: %d' % epoch +
                     ', Epochs on convergence counter: %d' % convergence)
        logging.info('Train: ')

        weight, train_loss, train_km_loss, train_kl_loss, train_mse_loss, fut_loss = train(train_loader, epoch, model, optimizer,
                                                                                           anneal_function, BETA, KL_START,
                                                                                           ANNEALTIME, TEMPORAL_WINDOW, FUTURE_DECODER,
                                                                                           FUTURE_STEPS, scheduler, MSE_REC_REDUCTION,
                                                                                           MSE_PRED_REDUCTION, KMEANS_LOSS, KMEANS_LAMBDA,
                                                                                           TRAIN_BATCH_SIZE, noise)

        test_mse_loss, test_loss, test_kmeans_loss = test(test_loader, epoch, model, optimizer,
                                                          BETA, weight, TEMPORAL_WINDOW, MSE_REC_REDUCTION,
                                                          KMEANS_LOSS, KMEANS_LAMBDA, FUTURE_DECODER, TEST_BATCH_SIZE)

        # logging losses
        train_losses.append(train_loss)
        train_kmeans_losses.append(train_km_loss)
        train_kl_losses.append(train_kl_loss)
        weight_values.append(weight)
        train_mse_losses.append(train_mse_loss)

        fut_losses.append(fut_loss)

        """fut_losses.append(fut_loss.cpu().item())"""

        test_losses.append(test_loss)
        test_mse_losses.append(test_mse_loss)
        test_kmeans_losses.append(test_kmeans_loss)

        lr = optimizer.param_groups[0]['lr']
        learn_rates.append(lr)

        lr_optimizer = optimizer.param_groups[0]['lr']

        wandb.log({
            'learning_rate': lr_optimizer,
            'train_loss': train_loss,
            'mse_loss_train': train_mse_loss,
            'test_loss': test_loss,
            'mse_loss_test': test_mse_loss,
            'kmeans_loss': train_km_loss,
            'kl_loss': train_kl_loss,
            'test_kmeans_loss': test_kmeans_loss,
            'weight': weight,
            'fut_loss': fut_loss,
            'epoch': epoch,
            'convergence': convergence,
            'max_epochs': EPOCHS,
            'zdims': ZDIMS,
            'beta': BETA,
            'model_snapshot': SNAPSHOT,
            'learning_rate': LEARNING_RATE,
            'num_features': NUM_FEATURES,
            'time_window': TEMPORAL_WINDOW,
            'prediction_decoder': FUTURE_DECODER,
            'prediction_steps': FUTURE_STEPS,
            'hidden_size_layer_1': hidden_size_layer_1,
            'hidden_size_layer_2': hidden_size_layer_2,
            'hidden_size_rec': hidden_size_rec,
            'hidden_size_pred': hidden_size_pred,
            'dropout_encoder': dropout_encoder,
            'dropout_rec': dropout_rec,
            'dropout_pred': dropout_pred,
            'noise': noise,
            'scheduler': optimizer_scheduler,
            'scheduler_step_size': scheduler_step_size,
            'scheduler_threshold': scheduler_thresh,
            'scheduler_gamma': scheduler_gamma,
            'softplus': softplus,
            'mse_reconstruction_reduction': MSE_REC_REDUCTION,
            'mse_prediction_reduction': MSE_PRED_REDUCTION,
            'kmeans_loss': KMEANS_LOSS,
            'kmeans_lambda': KMEANS_LAMBDA,
            'kl_start': KL_START,
            'annealtime': ANNEALTIME,
            'anneal_function': anneal_function,
            'batch_size': TRAIN_BATCH_SIZE
        })

        # save best model
        if weight > 0.99 and test_mse_loss <= BEST_LOSS:
            BEST_LOSS = test_mse_loss
            logging.info("Saving model!")
            torch.save(model.state_dict(), os.path.join(
                project_path, "model", "best_model", model_name+'_'+project+'.pkl'))
            convergence = 0
        else:
            convergence += 1
        conv_counter.append(convergence)

        # save model snapshot
        if epoch % SNAPSHOT == 0:
            logging.info("Saving model snapshot!\n")
            torch.save(model.state_dict(), os.path.join(project_path, 'model', 'best_model',
                       'snapshots', model_name+'_'+project+'_epoch_'+str(epoch)+'.pkl'))
            wandb.save(os.path.join(project_path, 'model', 'best_model',
                       'snapshots', model_name+'_'+project+'_epoch_'+str(epoch)+'.pkl'))

        # save logged losses
        np.save(os.path.join(project_path, 'model', 'model_losses',
                'train_losses_'+model_name+'_'+random_suffix), train_losses)
        np.save(os.path.join(project_path, 'model', 'model_losses',
                'test_losses_'+model_name+'_'+random_suffix), test_losses)
        np.save(os.path.join(project_path, 'model', 'model_losses',
                'kmeans_losses_'+model_name+'_'+random_suffix), train_kmeans_losses)
        np.save(os.path.join(project_path, 'model', 'model_losses',
                'kl_losses_'+model_name+'_'+random_suffix), train_kl_losses)
        np.save(os.path.join(project_path, 'model', 'model_losses',
                'weight_values_'+model_name+'_'+random_suffix), weight_values)
        np.save(os.path.join(project_path, 'model', 'model_losses',
                'mse_train_losses_'+model_name+'_'+random_suffix), train_mse_losses)
        np.save(os.path.join(project_path, 'model', 'model_losses',
                'mse_test_losses_'+model_name+'_'+random_suffix), test_mse_losses)

        np.save(os.path.join(project_path, 'model', 'model_losses',
                'fut_losses_' + model_name+'_'+random_suffix), fut_losses)

        """
        # Convert fut_losses to a tensor and save
        logging.debug("Converting fut_losses to a tensor and saving")
        fut_losses_tensor = torch.tensor(fut_losses)
        fut_losses_array = fut_losses_tensor.cpu().detach().numpy()
        np.save(os.path.join(project_path, 'model', 'model_losses', 'fut_losses_' + model_name+'_'+random_suffix), fut_losses_array)
        """
        # df_data = np.column_stack((train_losses, test_losses, train_kmeans_losses, train_kl_losses, weight_values, train_mse_losses, fut_losses, learn_rates, conv_counter))
        # df = pd.DataFrame(df_data.T, columns=['Train_losses', 'Test_losses', 'Kmeans_losses', 'KL_losses', 'Weight_values', 'MSE_losses', 'Future_losses', 'Learning_Rate', 'Convergence_counter'])
        df = pd.DataFrame([train_losses, test_losses, train_kmeans_losses, train_kl_losses,
                          weight_values, train_mse_losses, fut_losses, learn_rates, conv_counter]).T
        df.columns = ['Train_losses', 'Test_losses', 'Kmeans_losses', 'KL_losses',
                      'Weight_values', 'MSE_losses', 'Future_losses', 'Learning_Rate', 'Convergence_counter']
        df.to_csv(project_path+'/model/model_losses/'+model_name +
                  '_'+random_suffix+'_LossesSummary.csv')
        logging.info("\n")

        if convergence > model_convergence:
            logging.info('Finished training...')
            logging.info('Model converged. Please check your model with vame.evaluate_model(). \n'
                         'You can also re-run vame.trainmodel() to further improve your model. \n'
                         'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                         'and plug your current model name into _pretrained_model_. \n'
                         'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                         '\n'
                         'Next: \n'
                         'Use vame.pose_segmentation() to identify behavioral motifs in your dataset!')
            # return
            break

    if convergence < model_convergence:
        logging.info('Model seemed to have not reached convergence. You may want to check your model \n'
                     'with vame.evaluate_model(). If your satisfied you can continue with \n'
                     'Use vame.behavior_segmentation() to identify behavioral motifs!\n\n'
                     'OPTIONAL: You can re-run vame.rnn_model() to improve performance.')

    # Only stop wandb if the model converged or the max number of epochs was reached
    if convergence > model_convergence or epoch == EPOCHS:
        wandb.finish()


wandb.agent(sweep_id, function=train_model, count=10)
