import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from vame.util.auxiliary import read_config
from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE
from vame.model.rnn_vae import reconstruction_loss, future_reconstruction_loss, cluster_loss, kullback_leibler_loss, kl_annealing, gaussian



import lightning as L
from lightning import Trainer, LightningModule
from lightning import Trainer




class VAE(L.LightningModule):
     def __init__(self, hparams):
         super(VAE, self).__init__()
         self.hparams = hparams
         self.model = RNN_VAE(self.hparams['temporal_window'], self.hparams['zdims'], self.hparams['num_features'], self.hparams['future_decoder'], self.hparams['prediction_steps'], self.hparams['hidden_size_layer_1'], self.hparams['hidden_size_layer_2'], self.hparams['hidden_size_rec'], self.hparams['hidden_size_pred'], self.hparams['dropout_encoder'], self.hparams['dropout_rec'], self.hparams['dropout_pred'], self.hparams['softplus'])
         self.lr = self.hparams['learning_rate']
         self.BETA = self.hparams['beta']
         self.kl_start = self.hparams['kl_start']
         self.annealtime = self.hparams['annealtime']
         self.anneal_function = self.hparams['anneal_function']
         self.mse_red = self.hparams['mse_reconstruction_reduction']
         self.mse_pred = self.hparams['mse_prediction_reduction']
         self.kloss = self.hparams['kmeans_loss']
         self.klmbda = self.hparams['kmeans_lambda']
         self.bsize = self.hparams['batch_size']
         self.noise = self.hparams['noise']
         self.scheduler_step_size = self.hparams['scheduler_step_size']
         self.scheduler_thresh = self.hparams['scheduler_threshold']
         self.optimizer_scheduler = self.hparams['scheduler']
         self.train_losses = []
         self.test_losses = []
         self.kmeans_losses = []
         self.kl_losses = []
         self.weight_values = []
         self.mse_losses = []
         self.fut_losses = []
         self.learn_rates = []
         self.conv_counter = []
         self.best_loss = 999999
         self.convergence = 0
         self.seq_len = int(self.hparams['temporal_window']/2)
               


     def forward(self, x):
         return self.model(x)

     def training_step(self, batch, batch_idx, future_decoder):
         data_item = Variable(batch)
         data_item = data_item.permute(0,2,1)
    
         data = data_item[:,:int(self.hparams['temporal_window']/2),:].type('self.device')
         fut = data_item[:,int(self.hparams['temporal_window']/2):int(self.hparams['temporal_window']/2)+self.hparams['prediction_steps'],:].type('self.device')
         
         if self.noise == True:
             data_gaussian = gaussian(data,True,self.hparams['temporal_window']/2)
         else:
             data_gaussian = data
        
         if self.hparams['future_decoder']:
             data_tilde, future, latent, mu, logvar = self.model(data_gaussian)
             rec_loss = reconstruction_loss(data, data_tilde, self.mse_red)
             fut_rec_loss = future_reconstruction_loss(fut, future, self.mse_pred)
             kmeans_loss = cluster_loss(latent.T, self.kloss, self.klmbda, self.bsize)
             kl_loss = kullback_leibler_loss(mu, logvar)
             kl_weight = kl_annealing(self.current_epoch, self.kl_start, self.annealtime, self.anneal_function)
             loss = rec_loss + fut_rec_loss + self.BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
             self.fut_losses.append(fut_rec_loss.detach())
         else:
             data_tilde, latent, mu, logvar = self.model(data_gaussian)
             rec_loss = reconstruction_loss(data, data_tilde, self.mse_red)
             kl_loss = kullback_leibler_loss(mu, logvar)
             kmeans_loss = cluster_loss(latent.T, self.kloss, self.klmbda, self.bsize)
             kl_weight = kl_annealing(self.current_epoch, self.kl_start, self.annealtime, self.anneal_function)
             loss = rec_loss + self.BETA*kl_weight*kl_loss + kl_weight*kmeans_loss

         self.log('train_loss', loss)
         self.log('mse_loss', rec_loss)
         self.log('kl_loss', kl_loss)
         self.log('kmeans_loss', kmeans_loss)
         self.log('weight', kl_weight)
         if future_decoder:
             self.log('fut_loss', fut_rec_loss)

         self.train_losses.append(loss.item())
         self.mse_losses.append(rec_loss.item())
         self.kl_losses.append(kl_loss.item())
         self.kmeans_losses.append(kmeans_loss.item())
         self.weight_values.append(kl_weight)
         return {'kl_weight': kl_weight, 'loss': loss, 'mse_loss': rec_loss, 'kl_loss': kl_loss, 'kmeans_loss': kmeans_loss, 'fut_loss': fut_rec_loss}

     def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)
        self.log('avg_mse_loss', torch.stack([x['mse_loss'] for x in outputs]).mean())
        self.log('avg_kl_loss', torch.stack([x['kl_loss'] for x in outputs]).mean())
        self.log('avg_kmeans_loss', torch.stack([x['kmeans_loss'] for x in outputs]).mean())
        self.log('avg_fut_loss', torch.stack([x['fut_loss'] for x in outputs]).mean())
        self.log('avg_weight', torch.stack([x['kl_weight'] for x in outputs]).mean())

        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(avg_loss, torch.stack(self.mse_losses).mean(), self.BETA*self.weight_values[-1]*torch.stack(self.kl_losses).mean(), self.weight_values[-1]*torch.stack(self.kmeans_losses).mean(), self.weight_values[-1]))
        return {'loss': avg_loss, 'log': {'train_loss': avg_loss}}

     def validation_step(self, batch, batch_idx, kl_weight):
         data_item = Variable(batch)
         data_item = data_item.permute(0,2,1)
         seq_len_half = int(self.seq_len / 2)
         data = data_item[:,:seq_len_half,:].type('self.device').to()

         if self.hparams['future_decoder']:
             recon_images, _, latent, mu, logvar = self.model(data)
             rec_loss = reconstruction_loss(data, recon_images, self.mse_red)
             kl_loss = kullback_leibler_loss(mu, logvar)
             kmeans_loss = cluster_loss(latent.T, self.kloss, self.klmbda, self.bsize)
             loss = rec_loss + self.BETA*kl_weight*kl_loss+ kl_weight*kmeans_loss
         else:
             recon_images, latent, mu, logvar = self.model(data)
             rec_loss = reconstruction_loss(data, recon_images, self.mse_red)
             kl_loss = kullback_leibler_loss(mu, logvar)
             kmeans_loss = cluster_loss(latent.T, self.kloss, self.klmbda, self.bsize)
             loss = rec_loss + self.BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
         
         self.test_losses.append(loss.item())

         self.log('val_loss', loss)
         self.log('mse_loss', rec_loss)
         self.log('kl_loss', kl_loss)
         self.log('kmeans_loss', kmeans_loss)

     

         return {'val_loss': loss.item(), 'mse_loss': rec_loss.item(), 'kl_loss': kl_loss.item(), 'kmeans_loss': kmeans_loss}

     def validation_epoch_end(self, outputs):
         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
         avg_mse_loss = torch.stack([x['mse_loss'] for x in outputs]).mean()
         avg_kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
         avg_kmeans_loss = torch.stack([x['kmeans_loss'] for x in outputs]).mean()
         
         self.log('avg_mse_loss', avg_mse_loss)
         self.log('avg_kl_loss', avg_kl_loss)
         self.log('avg_kmeans_loss', avg_kmeans_loss)
         self.log('avg_val_loss', avg_loss)
        

     def configure_optimizers(self):
         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
         if self.optimizer_scheduler:
             self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.hparams['scheduler_gamma'], patience=self.hparams['scheduler_step_size'], threshold=self.scheduler_thresh, threshold_mode='rel', verbose=True)
         else:
             self.scheduler = StepLR(self.optimizer, step_size=self.scheduler_step_size, gamma=1, last_epoch=-1)
         return [self.optimizer], [self.scheduler]
     

     def train_dataloader(self):
         self.train_loader = Data.DataLoader(self.trainset, batch_size=self.hparams['batch_size'], shuffle=True, drop_last=True, num_workers=4)
         return self.train_loader

    # WHICH ONE IS RIGHT??
     def val_dataloader(self):
        self.test_loader = Data.DataLoader(self.testset, batch_size=int(self.hparams['batch_size']/4), shuffle=True, drop_last=True, num_workers=4)
        return self.test_loader

    # WHICH ONE IS RIGHT??
     def test_dataloader(self):
        self.test_loader = Data.DataLoader(self.testset, batch_size=int(self.hparams['batch_size']/4), shuffle=True, drop_last=True, num_workers=4)
        return self.test_loader

     def prepare_data(self):
        self.trainset = SEQUENCE_DATASET(os.path.join(self.hparams['project_path'],"data", "train",""), data='train_seq.npy', train=True, temporal_window=self.hparams['temporal_window'])
        self.testset = SEQUENCE_DATASET(os.path.join(self.hparams['project_path'],"data", "train",""), data='test_seq.npy', train=False, temporal_window=self.hparams['temporal_window'])
        return self.trainset, self.testset
     
# Start a Lighting Trainer class

callbacks = [EarlyStopping(monitor='val_loss'), ModelCheckpoint(filepath='model-{epoch:02d}-{val_loss:.2f}')]

trainer = L.Trainer(
    accelerator = ,
    gpus = ,
    precision = 16 ,
    batch_size = ,
    max_epochs = ,
    callbacks = ,
    progress_bar_refresh_rate = 1,
    profiler = ,


)
