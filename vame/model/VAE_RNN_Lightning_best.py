from pytorch_lightning import LightningDataModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch.nn.functional as F
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import vame.util.auxiliary as aux




# SEQUENCE_DATASET class remains the same
class SEQUENCE_DATASET(Dataset):
    def __init__(self,path_to_file,data,train,temporal_window):
        self.temporal_window = temporal_window        
        self.X = np.load(path_to_file+data)
        if self.X.shape[0] > self.X.shape[1]:
            self.X=self.X.T
            
        self.data_points = len(self.X[0,:])
        
        if train and not os.path.exists(os.path.join(path_to_file,'seq_mean.npy')):
            print("Compute mean and std for temporal dataset.")
            self.mean = np.mean(self.X)
            self.std = np.std(self.X)
            np.save(path_to_file+'seq_mean.npy', self.mean)
            np.save(path_to_file+'seq_std.npy', self.std)
        else:
            self.mean = np.load(path_to_file+'seq_mean.npy')
            self.std = np.load(path_to_file+'seq_std.npy')
        
        if train:
            print('Initialize train data. Datapoints %d' %self.data_points)
        else:
            print('Initialize test data. Datapoints %d' %self.data_points)
        
    def __len__(self):        
        return self.data_points

    def __getitem__(self, index):
        temp_window = self.temporal_window
        nf = self.data_points
        start = np.random.choice(nf-temp_window)
        end = start+temp_window
        sequence = self.X[:,start:end]  
        sequence = (sequence-self.mean)/self.std
        
        return torch.from_numpy(sequence)

"""DATA LOADER"""
class SequenceDataModule(LightningDataModule):
    def __init__(self, config, train_batch_size, test_batch_size, temporal_window):
        super(SequenceDataModule, self).__init__()
        self.config = config
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.temporal_window = temporal_window
        self.data_dir = os.path.join(self.config['project_path'], "data", "train", "")
        
    def prepare_data(self):
        # Here you might download the data
        pass
    
    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.trainset = SEQUENCE_DATASET(self.data_dir, data='train_seq.npy', train=True, temporal_window=self.temporal_window)
        if stage in (None, 'test'):
            self.testset = SEQUENCE_DATASET(self.data_dir, data='test_seq.npy', train=False, temporal_window=self.temporal_window)

    def train_dataloader(self):
        cuda_generator = torch.Generator(device='cuda') if torch.cuda.is_available() else None
        return DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=True, drop_last=True, num_workers=4, generator=cuda_generator)

    def val_dataloader(self):
        cuda_generator = torch.Generator(device='cuda') if torch.cuda.is_available() else None
        return DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=True, drop_last=True, num_workers=4, generator=cuda_generator)

"""nn.Modules"""
class Encoder(nn.Module):
    def __init__(self, NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder):
        super(Encoder, self).__init__()
        
        self.input_size = NUM_FEATURES
        self.hidden_size = hidden_size_layer_1
        self.hidden_size_2 = hidden_size_layer_2
        self.n_layers  = 2 
        self.dropout   = dropout_encoder
        self.bidirectional = True
        
        self.encoder_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)#UNRELEASED!
    
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers
        
    def forward(self, inputs):        
        outputs_1, hidden_1 = self.encoder_rnn(inputs)#UNRELEASED!
        
        hidden = torch.cat((hidden_1[0,...], hidden_1[1,...], hidden_1[2,...], hidden_1[3,...]),1)
        
        return hidden
    
    
class Lambda(nn.Module):
    def __init__(self,ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus):
        super(Lambda, self).__init__()
        
        self.hid_dim = hidden_size_layer_1*4
        self.latent_length = ZDIMS
        self.softplus = softplus
        
        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)
        
        if self.softplus == True:
            print("Using a softplus activation to ensures that the variance is parameterized as non-negative and activated by a smooth function")
            self.softplus_fn = nn.Softplus()
        
    def forward(self, hidden):
        
        self.mean = self.hidden_to_mean(hidden)
        if self.softplus == True:
            self.logvar = self.softplus_fn(self.hidden_to_logvar(hidden))
        else:
            self.logvar = self.hidden_to_logvar(hidden)
        
        if self.training:
            std = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.mean), self.mean, self.logvar
        else:
            return self.mean, self.mean, self.logvar

      
class Decoder(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec):
        super(Decoder,self).__init__()
        
        self.num_features = NUM_FEATURES
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_rec
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_rec
        self.bidirectional = True
        
        self.rnn_rec = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers # NEW
        
        self.latent_to_hidden = nn.Linear(self.latent_length,self.hidden_size * self.hidden_factor) # NEW
        self.hidden_to_output = nn.Linear(self.hidden_size*(2 if self.bidirectional else 1), self.num_features)
        
    def forward(self, inputs, z):
        batch_size = inputs.size(0) # NEW
        
        hidden = self.latent_to_hidden(z) # NEW
        
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size) # NEW
        
        decoder_output, _ = self.rnn_rec(inputs, hidden)
        prediction = self.hidden_to_output(decoder_output)
        
        return prediction
    
    
class Decoder_Future(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred, dropout_pred):
        super(Decoder_Future,self).__init__()
        
        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_pred
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_pred
        self.bidirectional = True
        
        self.rnn_pred = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers # NEW
        
        self.latent_to_hidden = nn.Linear(self.latent_length,self.hidden_size * self.hidden_factor)
        self.hidden_to_output = nn.Linear(self.hidden_size*2, self.num_features)
        
    def forward(self, inputs, z):
        batch_size = inputs.size(0)
        
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        
        inputs = inputs[:,:self.future_steps,:]
        decoder_output, _ = self.rnn_pred(inputs, hidden)
        
        prediction = self.hidden_to_output(decoder_output)
         
        return prediction


class LightningRNN_VAE(LightningModule):
    def __init__(self, config):
        super(LightningRNN_VAE, self).__init__()
        
        self.save_hyperparameters(config)

        # You can save the config as an attribute to access its elements later
        self.config = config

        # Initialize your original modules here
        self.encoder = Encoder(config['NUM_FEATURES'], config['hidden_size_layer_1'], 
                               config['hidden_size_layer_2'], config['dropout_encoder'])
        
        self.lmbda = Lambda(config['ZDIMS'], config['hidden_size_layer_1'], 
                            config['hidden_size_layer_2'], config['softplus'])
        
        self.decoder = Decoder(config['TEMPORAL_WINDOW'], config['ZDIMS'], 
                               config['NUM_FEATURES'], config['hidden_size_rec'], 
                               config['dropout_rec'])

        if config['FUTURE_DECODER']:
            self.decoder_future = Decoder_Future(config['TEMPORAL_WINDOW'], config['ZDIMS'], 
                                                 config['NUM_FEATURES'], config['FUTURE_STEPS'], 
                                                 config['hidden_size_pred'], config['dropout_pred'])
        
        # Initialize empty lists to store losses
        self.train_losses = []
        self.val_losses = []
        self.kmeans_losses = []
        self.kl_losses = []
        self.kl_weights = []
        self.mse_losses = []
        self.fut_losses = []
        self.learn_rates = []

        self.model_name = config['model_name']
        self.model_path = config['model_path']
        self.project_path = config['project_path']

        self.epoch = 0
    
        self.wandb = input("Do you want to use wandb for logging? (y/n): ")


    def forward(self, x):
        # Similar to your original forward method
        h_n = self.encoder(x)
        z, mu, logvar = self.lmbda(h_n)
        ins = z.unsqueeze(2).repeat(1, 1, self.config['TEMPORAL_WINDOW'] // 2)
        ins = ins.permute(0, 2, 1)
        prediction = self.decoder(ins, z)

        if self.config['FUTURE_DECODER']:
            future = self.decoder_future(ins, z)
            return prediction, future, z, mu, logvar
        else:
            return prediction, z, mu, logvar
        
    """Loss Functions"""

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
    
    
    
    def training_step(self, batch, batch_idx):
        # Extract data and move tensors to the appropriate device
        data_item = batch.to(self.device)
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

            rec_loss = self.reconstruction_loss(data, data_tilde, mse_red)
            fut_rec_loss = self.future_reconstruction_loss(fut, future, mse_pred)
            kmeans_loss = self.cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_loss = self.kullback_leibler_loss(mu, logvar)
            kl_weight = self.kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + fut_rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss
            fut_loss += fut_rec_loss.detach()#.item()

        else:
            data_tilde, latent, mu, logvar = model(data_gaussian)

            rec_loss = reconstruction_loss(data, data_tilde, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
            loss = rec_loss + BETA*kl_weight*kl_loss + kl_weight*kmeans_loss

        self.train_losses.append(loss.item())
        self.fut_losses.append(fut_loss.item())
        self.kmeans_losses.append(kmeans_loss.item())
        self.kl_losses.append(kl_loss.item())
        self.kl_weights.append(kl_weight)
        self.train_mse_losses.append(rec_loss.item())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        train_metrics = {'loss': loss, 'train_mse_loss': rec_loss, 'kl_loss': kl_loss, 'kl_weight': kl_weight, 'kmeans_loss': kmeans_loss, 'fut_loss': fut_loss}
        self.log_dict(train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_train_mse_loss = torch.stack([x['train_mse_loss'] for x in outputs]).mean()
        avg_train_kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        avg_train_kl_weight = torch.stack([x['kl_weight'] for x in outputs]).mean()
        avg_kmeans_loss = torch.stack([x['kmeans_loss'] for x in outputs]).mean()
        avg_train_fut_loss = torch.stack([x['fut_loss'] for x in outputs]).mean()
        
        train_epoch_metrics = {
            'avg_train_loss': avg_train_loss,
            'avg_train_mse_loss': avg_train_mse_loss,
            'avg_train_kl_loss': avg_train_kl_loss,
            'avg_train_kl_weight': avg_train_kl_weight,
            'avg_kmeans_loss': avg_kmeans_loss,
            'avg_train_fut_loss': avg_train_fut_loss
            }
    
        self.log_dict(train_epoch_metrics, prog_bar=True, logger=True)
        pass


    def validation_step(self, batch, batch_idx):
        data_item = batch.to(self.device)
        data_item = data_item.permute(0, 2, 1)
        seq_len_half = self.config['TEMPORAL_WINDOW'] // 2
        data = data_item[:, :seq_len_half, :]

        if self.config['FUTURE_DECODER']:
            recon_images, _, latent, mu, logvar = self(data)
            rec_loss = self.reconstruction_loss(data, recon_images)
            kl_loss = self.kullback_leibler_loss(mu, logvar)
            kmeans_loss = self.cluster_loss(latent.T)
            loss = rec_loss + self.config['BETA'] * self.current_kl_weight * kl_loss + self.current_kl_weight * kmeans_loss
        else:
            recon_images, latent, mu, logvar = self(data)
            rec_loss = self.reconstruction_loss(data, recon_images)
            kl_loss = self.kullback_leibler_loss(mu, logvar)
            kmeans_loss = self.cluster_loss(latent.T)
            loss = rec_loss + self.config['BETA'] * self.current_kl_weight * kl_loss + self.current_kl_weight * kmeans_loss

        self.val_losses.append(loss.item())
        self.val_mse_losses.append(rec_loss.item())
        self.kl_losses.append(kl_loss.item())
        self.kmeans_losses.append(kmeans_loss.item())
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        validation_metrics = {'val_loss': loss, 'val_mse_loss': rec_loss, 'kl_loss': kl_loss, 'kmeans_loss': kmeans_loss}
        self.log_dict(validation_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_mse_loss = torch.stack([x['val_mse_loss'] for x in outputs]).mean()
        avg_kl_loss = torch.stack([x['kl_loss'] for x in outputs]).mean()
        avg_kmeans_loss = torch.stack([x['val_kmeans_loss'] for x in outputs]).mean()
        
        self.epoch = self.epoch + 1

        test_epoch_metrics = {
            'avg_val_loss': avg_loss,
            'avg_val_mse_loss': avg_val_mse_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_kmeans_loss': avg_kmeans_loss
            }
        
        self.log_dict(test_epoch_metrics, prog_bar=True, on_epoch=True)
        pass

    def configure_optimizers(self):
        # Initializ optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['LEARNING_RATE'], amsgrad=True)
        
        lr = optimizer.param_groups[0]['lr']
        self.learn_rates = self.learn_rates.apend(lr)

        # Choose the scheduler based on your condition DON'T FORGET TO ADD THE SCHEDULER STEP SIZE TO THE CONFIG
        if self.config['optimizer_scheduler']:
            print(f"Scheduler step size: {self.config['scheduler_step_size']}, Scheduler gamma: {self.config['scheduler_gamma']}, Scheduler Threshold: {self.config['scheduler_threshold']}")
            scheduler = {
                'scheduler': ReduceLROnPlateau(optimizer, 'min', factor=self.config['scheduler_gamma'], 
                                                patience=self.config['scheduler_step_size'], 
                                                threshold=self.config['scheduler_threshold'], 
                                                threshold_mode='rel', verbose=True),
                'monitor': 'val_loss'  # The metric to monitor for ReduceLROnPlateau must be specified
            }
        else:
            scheduler = StepLR(optimizer, step_size=self.config['scheduler_step_size'], gamma=1, last_epoch=-1)
        
        # Return optimizer and scheduler
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def on_train_end(self):
        metrics_data = {
            'Train_losses': self.train_losses,
            'Val_losses': self.val_losses,
            'Fut_losses': self.fut_losses,
            'Kmeans_losses': self.kmeans_losses,
            'Test_losses': self.test_losses,
            'Train_MSE_losses': self.train_mse_losses,
            'Val_MSE_losses': self.val_mse_losses,
            'KL_losses': self.kl_losses,
            'KL_weights': self.kl_weights,
            'Learn_rates': self.learn_rates,
            }

        np.save(os.path.join(self.config['project_path'],'model','model_losses','train_losses_'+model_name), self.train_losses)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','val_losses_'+model_name), self.val_losses)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','kmeans_losses_'+model_name), self.kmeans_losses)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','test_losses_'+model_name), self.test_losses)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','train_mse_losses_'+model_name), self.train_mse_losses)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','val_mse_losses_'+model_name), self.val_mse_losses)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','kl_losses_'+model_name), self.kl_losses)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','kl_weights_'+model_name), self.kl_weights)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','learn_rates_'+model_name), self.learn_rates)
        np.save(os.path.join(self.config['project_path'],'model','model_losses','fut_losses_'+model_name), self.fut_losses)


        df = pd.DataFrame(metrics_data)
        df.columns = ['Train_losses', 'Val_losses', 'Fut_losses', 'Kmeans_losses', 'Test_losses', 'Train_MSE_losses', 'Val_MSE_losses', 'KL_losses', 'KL_weights', 'Learn_rates']
        df.to_csv(self.config['project_path']+'/model/model_losses/'+self.model_name+'_LossesSummary.csv')
    
    def initialize_wandb(self):
        wandb_logger = None  # Initialize to None as a default
        if self.wandb:
            self.wandb_username = input("Please enter your wandb username: ")
            self.wandb_project = input("Please enter your wandb project name: ")
            self.wandb_run_name = input("Please enter your wandb run name: ")
            self.wandb_api_key = getpass.getpass("Please enter your wandb api key: ")

            # Initialize logger
            wandb_logger = WandbLogger(
                username=self.wandb_username, 
                project=self.wandb_project, 
                name=self.wandb_run_name, 
                api_key=self.wandb_api_key
                )
            
            # Watch model
            wandb.watch(model)

        return wandb_logger




"""NEEDS WORK""""
import getpass

seed_everything(42)

# Your config and other parameters
config = {...} # Path to lightning_config file

# Initialize Model
model = LightningRNN_VAE(config)

# Initialize DataModule
data_module = SequenceDataModule(config, train_batch_size, test_batch_size, temporal_window)

# Intizialize Logger
wandb_logger = model.initialize_wandb()

# Initialize Checkpoint Callback
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(config['project_path'], 'model', 'best_model', 'snapshots'),  # Save in this directory
    filename=config["model_name"] + '_{epoch:02d}-{val_loss:.2f}',  # Filename format
    save_top_k=config["lightning_model_snapshots"],  # Save the top k models based on 'val_loss'
    mode=config["mode"],  # 'min' or 'max'
    patience=config["patience"],  # How many epochs to wait before stopping
    verbose=True,  # Output logs
    monitor='val_loss',  # Metric to monitor, OG VAME uses 'val_mse_loss'
    min_delta=config["min_delta"]  # Minimum change needed to qualify as an improvement
)

# Initialize Early Stopping Callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',  # Metric to monitor, OG VAME uses 'val_mse_loss'
    min_delta=config["min_delta"],  # Minimum change needed to qualify as an improvement
)


# Initialize Learning Rate Monitor Callback
lr_monitor = LearningRateMonitor(logging_interval='epoch')


# Initialize Trainer
trainer = Trainer(
    gpus=config["gpus"], # Number of gpus per node
    num_nodes=config["num_nodes"], # Number of nodes for distributed training
    accelerator='ddp',    # Distributed Data Parallel
    max_epochs=config["max_epochs"],
    batch_size=config["batch_size"],
    precision=config["precision"],
    progress_bar_refresh_rate=config["progress_bar_refresh_rate"],
    precision=config["precision"],
    distributed_backend=config["distributed_backend"],
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    auto_lr_find=config["auto_lr_find"],
    resume_from_checkpoint=config["resume_from_checkpoint"],
    ) 

# Training
trainer.fit(
    model=model,
    datamodule=data_module,
    train_dataloader=data_module.train_dataloader(),
    val_dataloader=data_module.val_dataloader()
    )

