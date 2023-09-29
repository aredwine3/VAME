# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0

The Model is partially adapted from the Timeseries Clustering repository developed by Tejas Lodaya:
https://github.com/tejaslodaya/timeseries-clustering-vae/blob/master/vrae/vrae.py
"""

import torch
from torch import nn
from torch.autograd import Variable

# NEW MODEL WITH SMALL ALTERATIONS
""" MODEL  """

class Encoder(nn.Module):
    """
    The main functionality of the Encoder class is to encode the input data into a latent space representation using a bidirectional 
    GRU (Gated Recurrent Unit) layer. It takes the input data and passes it through the GRU layer, concatenating the hidden states 
    from each layer and direction to obtain the encoded representation.
    """
    def __init__(self, NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder):
        """
        Initialize the Encoder module.

        Parameters:
        NUM_FEATURES (int): The number of features in the input data.
        hidden_size_layer_1 (int): The size of the first hidden layer.
        hidden_size_layer_2 (int): The size of the second hidden layer.
        dropout_encoder (float): The dropout rate for the encoder.
        """
        super(Encoder, self).__init__()
        
        self.input_size = NUM_FEATURES
        self.hidden_size = hidden_size_layer_1
        self.hidden_size_2 = hidden_size_layer_2
        self.n_layers  = 2 
        self.dropout   = dropout_encoder
        self.bidirectional = True
        
        # Define the GRU layer with the specified parameters
        self.encoder_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
    
        # Calculate the hidden factor based on the number of layers and the bidirectionality
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers
        
    def forward(self, inputs):        
        """
        Performs a forward pass of the Encoder module. It takes the input data and returns the encoded representation.

        Parameters:
        inputs (torch.Tensor): The input data.

        Returns:
        hidden (torch.Tensor): The encoded representation of the input data.
        """
        # Pass the input data through the GRU layer
        outputs_1, hidden_1 = self.encoder_rnn(inputs)
        
        # Concatenate the hidden states from each layer and direction
        hidden = torch.cat((hidden_1[0,...], hidden_1[1,...], hidden_1[2,...], hidden_1[3,...]),1)
        
        return hidden
    
    
class Lambda(nn.Module):
    """
    The Lambda class is a module in the RNN_VAE model that computes the latent state of the input sequence using the reparameterization trick.
    The reparameterization trick is a method used in variational autoencoders to allow backpropagation through random nodes.
    
    Args:
        ZDIMS (int): The dimensionality of the latent state. This is the size of the vector space in which the input data is embedded.
        hidden_size_layer_1 (int): The dimensionality of the hidden state. This is the size of the output vectors from the first layer of the GRU.
        hidden_size_layer_2 (int): Not used in this class.
        softplus (bool): A boolean indicating whether to use a softplus activation function for the log variance. The softplus function is a smooth approximation to the ReLU function.
    
    Attributes:
        hid_dim (int): The dimensionality of the hidden state. This is the size of the output vectors from the first layer of the GRU.
        latent_length (int): The dimensionality of the latent state. This is the size of the vector space in which the input data is embedded.
        softplus (bool): A boolean indicating whether to use a softplus activation function for the log variance. The softplus function is a smooth approximation to the ReLU function.
        hidden_to_mean (nn.Linear): A linear layer that maps the hidden state to the mean of the latent state. This is used to generate the mean vector for the Gaussian distribution from which we sample the latent state.
        hidden_to_logvar (nn.Linear): A linear layer that maps the hidden state to the log variance of the latent state. This is used to generate the log variance vector for the Gaussian distribution from which we sample the latent state.
        softplus_fn (nn.Softplus): A softplus activation function, used if `softplus` is True. The softplus function is a smooth approximation to the ReLU function.
    """
    
    def __init__(self, ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus):
        super(Lambda, self).__init__()
        
        # The dimensionality of the hidden state is four times the size of the output vectors from the first layer of the GRU.
        self.hid_dim = hidden_size_layer_1*4
        # The dimensionality of the latent state is the size of the vector space in which the input data is embedded.
        self.latent_length = ZDIMS
        # A boolean indicating whether to use a softplus activation function for the log variance.
        self.softplus = softplus
        
        # A linear layer that maps the hidden state to the mean of the latent state.
        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        # A linear layer that maps the hidden state to the log variance of the latent state.
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)
        
        # If the softplus flag is set to True, we use a softplus activation function for the log variance.
        if self.softplus == True:
            print("Using a softplus activation to ensures that the variance is parameterized as non-negative and activated by a smooth function")
            self.softplus_fn = nn.Softplus()
        
    def forward(self, hidden):
        """
        Computes the mean and log variance of the latent state given the hidden state of the encoder.
        If in training mode, it also samples from the latent space using the reparameterization trick.
        
        Args:
            hidden (torch.Tensor): The hidden state of the encoder.
        
        Returns:
            tuple: A tuple containing the latent state, mean, and log variance tensors. The latent state is a sampled point in the latent space, and the mean and log variance are the parameters of the Gaussian distribution from which the latent state was sampled.
        """
        # Compute the mean of the latent state.
        self.mean = self.hidden_to_mean(hidden)
        # Compute the log variance of the latent state. If the softplus flag is set to True, we apply the softplus function to the output of the linear layer.
        if self.softplus == True:
            self.logvar = self.softplus_fn(self.hidden_to_logvar(hidden))
        else:
            self.logvar = self.hidden_to_logvar(hidden)
        
        # If the model is in training mode, we sample a point in the latent space using the reparameterization trick.
        if self.training:
            # Compute the standard deviation from the log variance.
            std = torch.exp(0.5 * self.logvar)
            # Generate a random tensor with the same size as the standard deviation tensor.
            eps = torch.randn_like(std)
            # Sample a point in the latent space by adding the mean and the product of the random tensor and the standard deviation.
            return eps.mul(std).add_(self.mean), self.mean, self.logvar
        else:
            # If the model is not in training mode, we simply return the mean of the latent state.
            return self.mean, self.mean, self.logvar

      
class Decoder(nn.Module):
    """
    Decoder module that decodes the latent state into a sequence of predictions.
    
    Args:
        TEMPORAL_WINDOW (int): The length of the input sequence.
        ZDIMS (int): The dimensionality of the latent state.
        NUM_FEATURES (int): The number of features in the input sequence.
        hidden_size_rec (int): The size of the hidden layer in the GRU.
        dropout_rec (float): The dropout rate for the GRU.
    """
    def __init__(self, TEMPORAL_WINDOW, ZDIMS, NUM_FEATURES, hidden_size_rec, dropout_rec):
        super(Decoder, self).__init__()
        
        self.num_features = NUM_FEATURES
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_rec
        self.latent_length = ZDIMS
        self.n_layers = 1
        self.dropout = dropout_rec
        self.bidirectional = True
        
        self.rnn_rec = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                              bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size * self.hidden_factor)
        self.hidden_to_output = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.num_features)
        
    def forward(self, inputs, z):
        """
        Performs a forward pass of the Decoder module.
        
        Args:
            inputs (torch.Tensor): The input sequence.
            z (torch.Tensor): The latent state.
        
        Returns:
            torch.Tensor: The predicted output sequence.
        """
        batch_size = inputs.size(0)
        
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        
        decoder_output, _ = self.rnn_rec(inputs, hidden)
        prediction = self.hidden_to_output(decoder_output)
        
        return prediction
    
    
class Decoder_Future(nn.Module):
    """
    The Decoder_Future class is a module in the Variational Animal Motion Embedding (VAME) model that decodes the latent state into a sequence of predictions for future steps.
    It uses a bidirectional GRU (Gated Recurrent Unit) layer to generate these predictions.

    Args:
        TEMPORAL_WINDOW (int): The length of the input sequence.
        ZDIMS (int): The dimensionality of the latent state. This is the size of the vector space in which the input data is embedded.
        NUM_FEATURES (int): The number of features in the input sequence.
        FUTURE_STEPS (int): The number of future steps to predict.
        hidden_size_pred (int): The size of the hidden layer in the GRU.
        dropout_pred (float): The dropout rate for the GRU.

    Attributes:
        num_features (int): The number of features in the input sequence.
        future_steps (int): The number of future steps to predict.
        sequence_length (int): The length of the input sequence.
        hidden_size (int): The size of the hidden layer in the GRU.
        latent_length (int): The dimensionality of the latent state.
        n_layers (int): The number of layers in the GRU.
        dropout (float): The dropout rate for the GRU.
        bidirectional (bool): A boolean indicating whether the GRU is bidirectional.
        rnn_pred (nn.GRU): The GRU layer used for prediction.
        hidden_factor (int): A factor calculated based on the number of layers and the bidirectionality of the GRU.
        latent_to_hidden (nn.Linear): A linear layer that maps the latent state to the hidden state.
        hidden_to_output (nn.Linear): A linear layer that maps the hidden state to the output sequence.
    """

    def __init__(self, TEMPORAL_WINDOW, ZDIMS, NUM_FEATURES, FUTURE_STEPS, hidden_size_pred, dropout_pred):
        super(Decoder_Future, self).__init__()

        # Initialize the attributes with the provided arguments
        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_pred
        self.latent_length = ZDIMS
        self.n_layers = 1
        self.dropout = dropout_pred
        self.bidirectional = True

        # Define the GRU layer with the specified parameters
        self.rnn_pred = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                               bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        # Calculate the hidden factor based on the number of layers and the bidirectionality
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers

        # Define the linear layers for mapping the latent state to the hidden state and the hidden state to the output sequence
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size * self.hidden_factor)
        self.hidden_to_output = nn.Linear(self.hidden_size * 2, self.num_features)

    def forward(self, inputs, z):
        """
        Performs a forward pass of the Decoder_Future module. It takes the input sequence and the latent state, and returns the predicted output sequence for the future steps.

        Args:
            inputs (torch.Tensor): The input sequence.
            z (torch.Tensor): The latent state.

        Returns:
            torch.Tensor: The predicted output sequence for the future steps.
        """
        # Get the batch size from the input sequence
        batch_size = inputs.size(0)

        # Map the latent state to the hidden state
        hidden = self.latent_to_hidden(z)
        # Reshape the hidden state to match the expected input shape for the GRU
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        # Slice the input sequence to only include the future steps
        inputs = inputs[:, :self.future_steps, :]
        # Pass the sliced input sequence and the hidden state through the GRU
        decoder_output, _ = self.rnn_pred(inputs, hidden)

        # Map the hidden state to the output sequence
        prediction = self.hidden_to_output(decoder_output)

        return prediction


class RNN_VAE(nn.Module):
    """
    The RNN_VAE class is a module that combines the Encoder, Lambda, and Decoder modules to form a complete Variational Autoencoder (VAE) model.
    It also includes an optional Decoder_Future module for predicting future steps.

    Args:
        TEMPORAL_WINDOW (int): The length of the input sequence.
        ZDIMS (int): The dimensionality of the latent state. This is the size of the vector space in which the input data is embedded.
        NUM_FEATURES (int): The number of features in the input sequence.
        FUTURE_DECODER (bool): A flag indicating whether to include the Decoder_Future module for predicting future steps.
        FUTURE_STEPS (int): The number of future steps to predict.
        hidden_size_layer_1 (int): The size of the first hidden layer in the Encoder.
        hidden_size_layer_2 (int): The size of the second hidden layer in the Encoder.
        hidden_size_rec (int): The size of the hidden layer in the Decoder.
        hidden_size_pred (int): The size of the hidden layer in the Decoder_Future.
        dropout_encoder (float): The dropout rate for the Encoder.
        dropout_rec (float): The dropout rate for the Decoder.
        dropout_pred (float): The dropout rate for the Decoder_Future.
        softplus (bool): A flag indicating whether to use a softplus activation function for the log variance in the Lambda module.

    Attributes:
        FUTURE_DECODER (bool): A flag indicating whether to include the Decoder_Future module for predicting future steps.
        seq_len (int): The length of the input sequence divided by 2.
        encoder (Encoder): The Encoder module.
        lmbda (Lambda): The Lambda module.
        decoder (Decoder): The Decoder module.
        decoder_future (Decoder_Future, optional): The Decoder_Future module, if FUTURE_DECODER is True.
    """
    def __init__(self,TEMPORAL_WINDOW,
                 ZDIMS,
                 NUM_FEATURES,
                 FUTURE_DECODER,
                 FUTURE_STEPS,
                 hidden_size_layer_1, 
                 hidden_size_layer_2,
                 hidden_size_rec, 
                 hidden_size_pred, 
                 dropout_encoder, 
                 dropout_rec, 
                 dropout_pred, 
                 softplus):
        super(RNN_VAE,self).__init__()
        
        # Initialize the attributes with the provided arguments
        self.FUTURE_DECODER = FUTURE_DECODER
        self.seq_len = int(TEMPORAL_WINDOW / 2)
        self.encoder = Encoder(NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder)
        self.lmbda = Lambda(ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus)
        self.decoder = Decoder(self.seq_len,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec)
        if FUTURE_DECODER:
            self.decoder_future = Decoder_Future(self.seq_len,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred,
                                                 dropout_pred)
        
    def forward(self,seq):
        """
        Performs a forward pass of the RNN_VAE module.
        
        Args:
            seq (torch.Tensor): The input sequence.
        
        Returns:
            tuple: A tuple containing the predicted output sequence, the latent state, the mean, and the log variance tensors. If FUTURE_DECODER is True, it also includes the predicted output sequence for the future steps.
        """
        # Encode the input sequence into a hidden state
        h_n = self.encoder(seq)
        
        # Compute the latent state from the hidden state using the reparameterization trick
        z, mu, logvar = self.lmbda(h_n)
        # Repeat the latent state across the sequence length dimension and permute the dimensions to match the expected input shape for the Decoder
        ins = z.unsqueeze(2).repeat(1, 1, self.seq_len)
        ins = ins.permute(0,2,1)
        
        # Decode the latent state into a predicted output sequence
        prediction = self.decoder(ins, z)
        
        # If FUTURE_DECODER is True, decode the latent state into a predicted output sequence for the future steps
        if self.FUTURE_DECODER:
            future = self.decoder_future(ins, z)
            return prediction, future, z, mu, logvar
        else:
            return prediction, z, mu, logvar


#----------------------------------------------------------------------------------------
#                               LEGACY MODEL                                            |
#----------------------------------------------------------------------------------------


""" MODEL """
class Encoder_LEGACY(nn.Module):
    def __init__(self, NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder):
        super(Encoder_LEGACY, self).__init__()

        self.input_size = NUM_FEATURES
        self.hidden_size = hidden_size_layer_1
        self.hidden_size_2 = hidden_size_layer_2
        self.n_layers  = 1
        self.dropout   = dropout_encoder

        self.rnn_1 = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)

        self.rnn_2 = nn.GRU(input_size=self.hidden_size*2, hidden_size=self.hidden_size_2, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, inputs):
        outputs_1, hidden_1 = self.rnn_1(inputs)
        outputs_2, hidden_2 = self.rnn_2(outputs_1)

        h_n_1 = torch.cat((hidden_1[0,...], hidden_1[1,...]), 1)
        h_n_2 = torch.cat((hidden_2[0,...], hidden_2[1,...]), 1)

        h_n = torch.cat((h_n_1, h_n_2), 1)

        return h_n


class Lambda_LEGACY(nn.Module):
    def __init__(self,ZDIMS, hidden_size_layer_1, hidden_size_layer_2):
        super(Lambda_LEGACY, self).__init__()

        self.hid_dim = hidden_size_layer_1*2 + hidden_size_layer_2*2
        self.latent_length = ZDIMS

        self.hidden_to_linear = nn.Linear(self.hid_dim, self.hid_dim)
        self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)

        self.softplus = nn.Softplus()

    def forward(self, cell_output):
        self.latent_mean = self.hidden_to_mean(cell_output)

        # based on Pereira et al 2019:
        # "The SoftPlus function ensures that the variance is parameterized as non-negative and activated
        # by a smooth function
        self.latent_logvar = self.softplus(self.hidden_to_logvar(cell_output))

        if self.training:
            std = self.latent_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(self.latent_mean), self.latent_mean, self.latent_logvar
        else:
            return self.latent_mean, self.latent_mean, self.latent_logvar


class Decoder_LEGACY(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec):
        super(Decoder_LEGACY,self).__init__()

        self.num_features = NUM_FEATURES
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_rec
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_rec

        self.rnn_rec = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=False)

        self.hidden_to_output = nn.Linear(self.hidden_size, self.num_features)

    def forward(self, inputs):
        decoder_output, _ = self.rnn_rec(inputs)
        prediction = self.hidden_to_output(decoder_output)

        return prediction

class Decoder_Future_LEGACY(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred, dropout_pred):
        super(Decoder_Future_LEGACY,self).__init__()

        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_pred
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_pred

        self.rnn_pred = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=True)

        self.hidden_to_output = nn.Linear(self.hidden_size*2, self.num_features)

    def forward(self, inputs):
        inputs = inputs[:,:self.future_steps,:]
        decoder_output, _ = self.rnn_pred(inputs)
        prediction = self.hidden_to_output(decoder_output)

        return prediction


class RNN_VAE_LEGACY(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                        hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                        dropout_rec, dropout_pred, softplus):
        super(RNN_VAE_LEGACY,self).__init__()

        self.FUTURE_DECODER = FUTURE_DECODER
        self.seq_len = int(TEMPORAL_WINDOW / 2)
        self.encoder = Encoder_LEGACY(NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder)
        self.lmbda = Lambda_LEGACY(ZDIMS, hidden_size_layer_1, hidden_size_layer_2)
        self.decoder = Decoder_LEGACY(self.seq_len,ZDIMS,NUM_FEATURES, hidden_size_rec, dropout_rec)
        if FUTURE_DECODER:
            self.decoder_future = Decoder_Future_LEGACY(self.seq_len,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred,
                                                 dropout_pred)

    def forward(self,seq):

        """ Encode input sequence """
        h_n = self.encoder(seq)

        """ Compute the latent state via reparametrization trick """
        latent, mu, logvar = self.lmbda(h_n)
        z = latent.unsqueeze(2).repeat(1, 1, self.seq_len)
        z = z.permute(0,2,1)

        """ Predict the future of the sequence from the latent state"""
        prediction = self.decoder(z)

        if self.FUTURE_DECODER:
            future = self.decoder_future(z)
            return prediction, future, latent, mu, logvar
        else:
            return prediction, latent, mu, logvar
        
