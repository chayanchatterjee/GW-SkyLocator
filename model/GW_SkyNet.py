# -*- coding: utf-8 -*-
"""Normalizing Flow model"""

''' 
 * Copyright (C) 2021 Chayan Chatterjee <chayan.chatterjee@research.uwa.edu.au>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
'''

# standard library

# internal

from .base_model import BaseModel
from dataloader.dataloader import DataLoader
from .wavenet import WaveNet
from .resnet import ResNet
from .resnet_34 import ResNet34
from .resnet_34_2_det import ResNet_34_2_det
from .lstm import LSTM_model
from .cnn import CNN_model
from utils.custom_checkpoint import CustomCheckpoint 

# external
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np
import math
import pandas as pd
import seaborn as sns

from scipy import stats
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors
import random
import os

import healpy as hp
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

# GPU specifications

device_type = 'GPU'
n_gpus = 2
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])
#try:
#    tf.config.experimental.set_memory_growth(devices[1], True)
#except:
  # Invalid device or cannot modify virtual devices once initialized.
#    pass


class GW_SkyNet(BaseModel):
    """Normalizing Flow Model Class"""
    def __init__(self, config):
        
        super().__init__(config)
        
        self.network = self.config.train.network
        self.dataset = self.config.train.dataset
        self.X_train_real = []
        self.X_train_imag = []
        self.X_test_real = []
        self.X_test_imag = []
        self.y_train = []
        self.y_test = []
        self.ra = []
        self.ra_x = []
        self.ra_y = []
        self.dec = []
        self.ra_test = []
        self.ra_test_x = []
        self.ra_test_y = []
        self.dec_test = []

        self.encoded_features = None
        self.model = None
        self.encoder = None
        self.sc = None
        
        self.num_train = self.config.train.num_train
        self.num_test = self.config.train.num_test
        self.n_samples = self.config.train.n_samples
        self.test_real = self.config.train.test_real
        self.snr_range_train = self.config.train.snr_range_train
        self.snr_range_test = self.config.train.snr_range_test
        self.min_snr = self.config.train.min_snr
        self.n_det = self.config.train.num_detectors
        self.epochs = self.config.train.epochs
        self.lr = self.config.model.learning_rate
        self.batch_size = self.config.train.batch_size
        self.val_split = self.config.train.validation_split
        self.output_filename = self.config.train.output_filename
        
        # Reading parameters specific to a neural network model  
        if(self.network == 'WaveNet'):
            self.filters = self.config.model.WaveNet.filters
            self.kernel_size = self.config.model.WaveNet.kernel_size
            self.activation = self.config.model.WaveNet.activation
            self.dilation_rate = self.config.model.WaveNet.dilation_rate
            
        elif(self.network == 'ResNet'):
            self.kernels_res = self.config.model.ResNet.kernels_resnet_block
            self.stride_res = self.config.model.ResNet.stride_resnet_block
            self.kernel_size_res = self.config.model.ResNet.kernel_size_resnet_block
            self.kernels = self.config.model.ResNet.kernels
            self.kernel_size = self.config.model.ResNet.kernel_size
            self.strides = self.config.model.ResNet.strides
            
        elif(self.network == 'ResNet-34'):
            self.filters = self.config.model.ResNet_34.filters
            self.kernel_size = self.config.model.ResNet_34.kernel_size
            self.strides = self.config.model.ResNet_34.strides
            self.pool_size = self.config.model.ResNet_34.pool_size
            self.prev_filters = self.config.model.ResNet_34.prev_filters
            
        elif(self.network == 'ResNet-34_2_det'):
            self.filters = self.config.model.ResNet_34_2_det.filters
            self.kernel_size = self.config.model.ResNet_34_2_det.kernel_size
            self.strides = self.config.model.ResNet_34_2_det.strides
            self.pool_size = self.config.model.ResNet_34_2_det.pool_size
            self.prev_filters = self.config.model.ResNet_34_2_det.prev_filters
            
        elif(self.network == 'LSTM'):
            self.n_units = self.config.model.LSTM_model.n_units
            self.rate = self.config.model.LSTM_model.rate
            
        elif(self.network == 'CNN'):
            self.filters = self.config.model.CNN_model.filters
            self.kernel_size = self.config.model.CNN_model.kernel_size
            self.max_pool_size = self.config.model.CNN_model.max_pool_size
            self.rate = self.config.model.CNN_model.dropout_rate
            self.n_units = self.config.model.CNN_model.n_units

        self.num_bijectors = self.config.model.num_bijectors
        self.trainable_distribution = None
        self.MAF_hidden_units = self.config.model.MAF_hidden_units
        
    def load_data(self):
        """Loads real and imaginary parts of SNR time series data """
        
        d_loader = DataLoader(self.n_det, self.dataset, self.num_test, self.n_samples, self.min_snr)
        
        if(self.n_det == 3):
            self.X_train_real, self.X_train_imag = d_loader.load_train_3_det_data(self.config.data, self.snr_range_train)
            self.X_test_real, self.X_test_imag = d_loader.load_test_3_det_data(self.config.data, self.test_real, self.snr_range_test)
                
            self.y_train = d_loader.load_train_3_det_parameters(self.config.parameters, self.snr_range_train)
            self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test = d_loader.load_test_3_det_parameters(self.config.parameters, self.test_real, self.snr_range_test)
            
        elif(self.n_det == 2):
            self.X_train_real, self.X_train_imag = d_loader.load_train_2_det_data(self.config.data, self.snr_range_train)
            self.X_test_real, self.X_test_imag = d_loader.load_test_2_det_data(self.config.data, self.test_real, self.snr_range_test)
                
            self.y_train = d_loader.load_train_2_det_parameters(self.config.parameters, self.snr_range_train)
            self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test = d_loader.load_test_2_det_parameters(self.config.parameters, self.test_real, self.snr_range_test)
        
        self._preprocess_data(d_loader)
        
    def _preprocess_data(self, d_loader):
        """ Removing samples with single detector SNR < self.min_snr. Normalizing amplitudes of SNR time series signals """
        
        if(self.n_det == 3):
            self.X_train_real, self.X_train_imag, self.y_train, self.ra_x, self.ra_y, self.ra, self.dec, self.h1_snr, self.l1_snr, self.v1_snr = d_loader.load_3_det_samples(self.config.parameters, self.X_train_real, self.X_train_imag, self.y_train, self.num_train, self.snr_range_train, self.snr_range_test, data='train')
            
            # Testing on real events/injections
            if(self.test_real == False):
                self.X_test_real, self.X_test_imag, self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test, self.h1_snr_test, self.l1_snr_test, self.v1_snr_test = d_loader.load_3_det_samples(self.config.parameters, self.X_test_real, self.X_test_imag, self.y_test, self.num_test, self.snr_range_train, self.snr_range_test, data='test')
            
            elif(self.test_real == True):
                self.X_test_real, self.X_test_imag, self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test, self.h1_snr_test, self.l1_snr_test, self.v1_snr_test = d_loader.load_3_det_samples(self.config.parameters, self.X_test_real, self.X_test_imag, self.y_test, self.num_test, self.snr_range_train, self.snr_range_test, data='real_event')
                            
            h1_real_train_scaled = []
            l1_real_train_scaled = []
            v1_real_train_scaled = []

            h1_imag_train_scaled = []
            l1_imag_train_scaled = []
            v1_imag_train_scaled = []

            for i in range(self.num_train):
                
                sc_h1_real = MinMaxScaler()
                sc_h1_imag = MinMaxScaler()

                h1_real_train_scaled.append(sc_h1_real.fit_transform(self.X_train_real[i,:,0].reshape(-1,1)))
                l1_real_train_scaled.append(sc_h1_real.transform(self.X_train_real[i,:,1].reshape(-1,1)))
                v1_real_train_scaled.append(sc_h1_real.transform(self.X_train_real[i,:,2].reshape(-1,1)))
    
                h1_imag_train_scaled.append(sc_h1_imag.fit_transform(self.X_train_imag[i,:,0].reshape(-1,1)))
                l1_imag_train_scaled.append(sc_h1_imag.transform(self.X_train_imag[i,:,1].reshape(-1,1)))
                v1_imag_train_scaled.append(sc_h1_imag.transform(self.X_train_imag[i,:,2].reshape(-1,1)))
            
                            
            h1_real_train_scaled = np.array(h1_real_train_scaled)
            h1_imag_train_scaled = np.array(h1_imag_train_scaled)

            l1_real_train_scaled = np.array(l1_real_train_scaled)
            l1_imag_train_scaled = np.array(l1_imag_train_scaled)
            
            v1_real_train_scaled = np.array(v1_real_train_scaled)
            v1_imag_train_scaled = np.array(v1_imag_train_scaled)
            
            self.X_train_real = np.concatenate([h1_real_train_scaled, l1_real_train_scaled, v1_real_train_scaled], axis=-1)
            self.X_train_imag = np.concatenate([h1_imag_train_scaled, l1_imag_train_scaled, v1_imag_train_scaled], axis=-1)

            
            h1_real_test_scaled = []
            l1_real_test_scaled = []
            v1_real_test_scaled = []

            h1_imag_test_scaled = []
            l1_imag_test_scaled = []
            v1_imag_test_scaled = []

            for i in range(self.num_test):
                
                sc_h1_real = MinMaxScaler()
                sc_h1_imag = MinMaxScaler()

                h1_real_test_scaled.append(sc_h1_real.fit_transform(self.X_test_real[i,:,0].reshape(-1,1)))
                l1_real_test_scaled.append(sc_h1_real.transform(self.X_test_real[i,:,1].reshape(-1,1)))
                v1_real_test_scaled.append(sc_h1_real.transform(self.X_test_real[i,:,2].reshape(-1,1)))
    
                h1_imag_test_scaled.append(sc_h1_imag.fit_transform(self.X_test_imag[i,:,0].reshape(-1,1)))
                l1_imag_test_scaled.append(sc_h1_imag.transform(self.X_test_imag[i,:,1].reshape(-1,1)))
                v1_imag_test_scaled.append(sc_h1_imag.transform(self.X_test_imag[i,:,2].reshape(-1,1)))
                                       
            h1_real_test_scaled = np.array(h1_real_test_scaled)
            h1_imag_test_scaled = np.array(h1_imag_test_scaled)

            l1_real_test_scaled = np.array(l1_real_test_scaled)
            l1_imag_test_scaled = np.array(l1_imag_test_scaled)
            
            v1_real_test_scaled = np.array(v1_real_test_scaled)
            v1_imag_test_scaled = np.array(v1_imag_test_scaled)
            
            self.X_test_real = np.concatenate([h1_real_test_scaled, l1_real_test_scaled, v1_real_test_scaled], axis=-1)
            self.X_test_imag = np.concatenate([h1_imag_test_scaled, l1_imag_test_scaled, v1_imag_test_scaled], axis=-1)
            
        
        elif(self.n_det == 2):
            
            self.X_train_real, self.X_train_imag, self.y_train, self.ra_x, self.ra_y, self.ra, self.dec, self.h1_snr, self.l1_snr = d_loader.load_2_det_samples(self.config.parameters, self.X_train_real, self.X_train_imag, self.y_train, self.num_train, self.snr_range_train, self.snr_range_test, data='train')
            
            self.X_test_real, self.X_test_imag, self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test, self.h1_snr_test, self.l1_snr_test = d_loader.load_2_det_samples(self.config.parameters, self.X_test_real, self.X_test_imag, self.y_test, self.num_test, self.snr_range_train, self.snr_range_test, data='test')
            
            h1_real_train_scaled = []
            l1_real_train_scaled = []

            h1_imag_train_scaled = []
            l1_imag_train_scaled = []

            for i in range(len(self.X_train_real)):
                
                sc_h1_real = MinMaxScaler()
                sc_h1_imag = MinMaxScaler()

                h1_real_train_scaled.append(sc_h1_real.fit_transform(self.X_train_real[i,:,0].reshape(-1,1)))
                l1_real_train_scaled.append(sc_h1_real.transform(self.X_train_real[i,:,1].reshape(-1,1)))
    
                h1_imag_train_scaled.append(sc_h1_imag.fit_transform(self.X_train_imag[i,:,0].reshape(-1,1)))
                l1_imag_train_scaled.append(sc_h1_imag.transform(self.X_train_imag[i,:,1].reshape(-1,1)))
            
                            
            h1_real_train_scaled = np.array(h1_real_train_scaled)
            h1_imag_train_scaled = np.array(h1_imag_train_scaled)

            l1_real_train_scaled = np.array(l1_real_train_scaled)
            l1_imag_train_scaled = np.array(l1_imag_train_scaled)
            
            self.X_train_real = np.concatenate([h1_real_train_scaled, l1_real_train_scaled], axis=-1)
            self.X_train_imag = np.concatenate([h1_imag_train_scaled, l1_imag_train_scaled], axis=-1)

            h1_real_test_scaled = []
            l1_real_test_scaled = []

            h1_imag_test_scaled = []
            l1_imag_test_scaled = []

            for i in range(len(self.X_test_real)):
                
                sc_h1_real = MinMaxScaler()
                sc_h1_imag = MinMaxScaler()

                h1_real_test_scaled.append(sc_h1_real.fit_transform(self.X_test_real[i,:,0].reshape(-1,1)))
                l1_real_test_scaled.append(sc_h1_real.transform(self.X_test_real[i,:,1].reshape(-1,1)))
    
                h1_imag_test_scaled.append(sc_h1_imag.fit_transform(self.X_test_imag[i,:,0].reshape(-1,1)))
                l1_imag_test_scaled.append(sc_h1_imag.transform(self.X_test_imag[i,:,1].reshape(-1,1)))
                                       
            h1_real_test_scaled = np.array(h1_real_test_scaled)
            h1_imag_test_scaled = np.array(h1_imag_test_scaled)

            l1_real_test_scaled = np.array(l1_real_test_scaled)
            l1_imag_test_scaled = np.array(l1_imag_test_scaled)
            
            self.X_test_real = np.concatenate([h1_real_test_scaled, l1_real_test_scaled], axis=-1)
            self.X_test_imag = np.concatenate([h1_imag_test_scaled, l1_imag_test_scaled], axis=-1)
            
            
        self.X_train_real = self.X_train_real.astype("float32")
        self.X_train_imag = self.X_train_imag.astype("float32")
        self.y_train = self.y_train.astype("float32")

        self.X_test_real = self.X_test_real.astype("float32")
        self.X_test_imag = self.X_test_imag.astype("float32")
        self.y_test = self.y_test.astype("float32")
        
    def construct_model(self):
        """ Constructing the neural network encoder model
        
        Args:
            model_type:     'wavenet', 'resnet', 'resnet-34', 'lstm', 'cnn'
            
            kwargs:         Depends on the model_type
            
                'wavenet'   input_dim_real  [n_samples, n_detectors]
                            input_dim_imag  [n_samples, n_detectors]
                            filters         Number of filters in each layer
                            kernel_size     Size of kernel in each layer
                            activation      (relu)
                            dilation_rate   Initial dilation rate for CNN layers
                            
                'resnet'    input_dim_real   [n_samples, n_detectors]
                            input_dim_imag   [n_samples, n_detectors]
                            kernels_res      Number of kernels in ResNet block
                            stride_res       Stride in ResNet block
                            kernel_size_res  Kernel size in ResNet block
                            kernels          Number of kernels in CNN layers
                            kernel_size      Kernel size in CNN layers
                            strides          Strides in CNN layers
               
               'resnet-34'  input_dim_real   [n_samples, n_detectors]
                            input_dim_imag   [n_samples, n_detectors]
                            filters          Number of filters in main layer
                            kernel_size      Kernel size in main layer
                            strides          Strides in main layer
                            prev_filters     Number of filters in previous main/Residual layer
                            input_shapes     Shapes of input signals
                            
               
               'lstm'       input_dim_real   [None, n_detectors]
                            input_dim_imag   [None, n_detectors]
                            units            Number of neurons
                            rate             Dropout rate
                            
               'cnn'        input_dim_real   [n_samples, n_detectors]
                            input_dim_imag   [n_samples, n_detectors]
                            filters          Numer of filters in CNN layer
                            kernel_size      Size of filters in convolutional layer
                            max_pool_size    Factor by which the size of feature maps are reduced
                            rate             Dropout rate
                            n_units          Number of neurons in fully connected layers
            
            returns:        A vector of encoded features extracted from the SNR times series data                
        
                         
        """

        with strategy.scope():
            
            input1 = tf.keras.layers.Input([self.n_samples, self.n_det])
            input2 = tf.keras.layers.Input([self.n_samples, self.n_det])
            
            if(self.network == 'WaveNet'):
                
                filters = self.filters
                kernel_size = self.kernel_size
                activation = self.activation
                dilation_rate = self.dilation_rate
                
                self.encoded_features = WaveNet(input1, input2, self.filters, self.kernel_size, self.activation, self.dilation_rate).construct_model()
            
            elif(self.network == 'ResNet'):
                
                self.encoded_features = ResNet(input1, input2, self.kernels_res, self.kernel_size_res, self.stride_res,
                                               self.kernels, self.kernel_size, self.strides).construct_model()
            elif(self.network == 'ResNet-34'):
                
                self.encoded_features = ResNet34(input1, input2, self.filters, self.kernel_size, self.strides, self.pool_size, self.prev_filters, input_shapes=[self.n_samples, self.n_det]).construct_model()
                
            elif(self.network == 'ResNet-34_2_det'):
                
                self.encoded_features = ResNet_34_2_det(input1, input2, self.filters, self.kernel_size, self.strides, self.pool_size, self.prev_filters, input_shapes=[self.n_samples, self.n_det]).construct_model()
                
            elif(self.network == 'LSTM'):
                
                self.encoded_features = LSTM_model(input1, input2, self.n_units, self.rate, input_shapes=[None, self.n_det]).construct_model()
                
            elif(self.network == 'CNN'):
                
                self.encoded_features = CNN_model(input1, input2, self.filters, self.kernel_size, self.max_pool_size, self.rate, self.n_units, input_shapes=[self.n_samples, self.n_det]).construct_model()
                
            x_ = tf.keras.layers.Input(shape=self.y_train.shape[-1], dtype=tf.float32)
        
            # Constructing the chain of Masked Autoregressive Flow bijectors
            bijectors = []

            for i in range(self.num_bijectors):
#                bijectors.append(tfb.BatchNormalization(name='batch_normalization'+str(i))) 
                masked_auto_i = self.make_masked_autoregressive_flow(i, hidden_units = self.MAF_hidden_units, activation = 'relu', conditional_event_shape=self.encoded_features.shape[-1])
                
                bijectors.append(masked_auto_i)
    
                USE_BATCHNORM = True
    
                if USE_BATCHNORM and i % 2 == 0:
                # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
                    bijectors.append(tfb.BatchNormalization(name='batch_normalization'+str(i)))

#                bijectors.append(tfb.BatchNormalization(name='batch_normalization'+str(i)))
    
                bijectors.append(tfb.Permute(permutation = [2, 1, 0]))
                flow_bijector = tfb.Chain(list(reversed(bijectors[:-1]))) # The chain of MAF bijectors is defined
            
            # Define the trainable distribution
            self.trainable_distribution = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(3)),
                        bijector = flow_bijector)
            
            # Defining the log likelihood loss
            log_prob_ = self.trainable_distribution.log_prob(x_, bijector_kwargs=
                        self.make_bijector_kwargs(self.trainable_distribution.bijector, 
                                             {'maf.': {'conditional_input':self.encoded_features}}))

            self.model = tf.keras.Model([input1, input2, x_], log_prob_)
            self.encoder = tf.keras.Model([input1, input2], self.encoded_features)  
  
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)  # optimizer
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=self.model)
            
            # load best model with min validation loss
#            checkpoint.restore('/fred/oz016/Chayan/GW-SkyNet/checkpoints/BNS_2_det_ResNet_adaptive/tmp_0xf6404635/ckpt-1')
#            checkpoint.restore('/home/cchatterjee/GW-SkyNet/checkpoints/BBH_2_det_ResNet-34_adaptive/tmp_0x280e68c2/ckpt-1')
#            self.encoder.load_weights("/fred/oz016/Chayan/GW-SkyNet/model/encoder_models/ResNet_BNS_encoder_2_det_adaptive_snr-10to20.hdf5")

        self.train(log_prob_, checkpoint)
    

    # Define the trainable distribution
    def make_masked_autoregressive_flow(self, index, hidden_units, activation, conditional_event_shape):
    
        made = tfp.bijectors.AutoregressiveNetwork(params=2,
              hidden_units=hidden_units,
              event_shape=(3,),
              activation=activation,
              conditional=True,
#              kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg', distribution='uniform'),
              kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1),
#              kernel_initializer = tf.keras.initializers.LecunNormal(),
              conditional_event_shape=conditional_event_shape)
    
        return tfp.bijectors.MaskedAutoregressiveFlow(shift_and_log_scale_fn = made, name='maf'+str(index))

    def make_bijector_kwargs(self, bijector, name_to_kwargs):
        
        if hasattr(bijector, 'bijectors'):
            
            return {b.name: self.make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    
        else:
            for name_regex, kwargs in name_to_kwargs.items():
                if re.match(name_regex, bijector.name):
                    return kwargs
        return {}
    
    
    def train(self, log_prob, checkpoint):
        """Compiles and trains the model"""
        
        custom_checkpoint = CustomCheckpoint(filepath='/fred/oz016/Chayan/GW-SkyNet/model/encoder_models/'+str(self.network)+'_'+str(self.dataset)+'_encoder_'+str(self.n_det)+'_det_adaptive_snr-10to20.hdf5',encoder=self.encoder)
        
        self.model.compile(optimizer=tf.optimizers.Adam(lr=self.lr), loss=lambda _, log_prob: -log_prob)
        self.model.summary()
                                             
        # initialize checkpoints
        dataset_name = "/fred/oz016/Chayan/GW-SkyNet/checkpoints/"+str(self.dataset)+"_"+str(self.n_det)+"_det_"+str(self.network)+"_adaptive"
        checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

        callbacks_list=[custom_checkpoint]  
        
        # Fitting the model to the training data
        self.model.fit([self.X_train_real, self.X_train_imag, self.y_train], np.zeros((len(self.X_train_real), 0), dtype=np.float32),
              batch_size=self.batch_size,
              epochs=self.epochs,
              validation_split=self.val_split,
              callbacks=callbacks_list,
              shuffle=True,
              verbose=True)

        checkpoint.save(file_prefix=checkpoint_prefix)
     
    # Defining the function to calculate 2D Kernel Density Estimation for samples drawn from the trained distribution.      
    def kde2D(self, x, y, bandwidth, ra_pix, de_pix, xbins=150j, ybins=150j, **kwargs):
        """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
#    xx, yy = np.mgrid[x.min():x.max():xbins, 
#                      y.min():y.max():ybins]

        xy_sample = np.vstack([de_pix, ra_pix]).T
        xy_train  = np.vstack([y, x]).T

        kde_skl = KernelDensity(kernel='gaussian', bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train)
    #    gm = GaussianMixture(n_components=10, random_state=0).fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        z = np.exp(kde_skl.score_samples(xy_sample))
        return z

    # Draw samples from the trained distribution         
    def obtain_samples(self):
        """Obtain samples from trained distribution"""
        self.encoder.load_weights('/fred/oz016/Chayan/GW-SkyNet/model/encoder_models/'+str(self.network)+'_'+str(self.dataset)+'_encoder_'+str(self.n_det)+'_det_adaptive_snr-10to20.hdf5')
        n_samples = 5000
        probs = []
        ra_preds = []
        dec_preds = []
        
        # Defining fixed resolution of sky grids. Adaptive Mesh Refinement can be used for the same but is slower. Needs speed-up using GPU acceleration
        nside=32
        npix=hp.nside2npix(nside)
        theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))

        # ra_pix and de_pix are co-ordinates in the sky where I want to find the probabilities
        ra_pix = phi
        de_pix = -theta + np.pi/2.0 

        for i in range(self.y_test.shape[0]):
            x_test_real = np.expand_dims(self.X_test_real[i],axis=0)
            x_test_imag = np.expand_dims(self.X_test_imag[i],axis=0)
            
            # Predicting the encoded feature vector from the saved encoder model
            preds = self.encoder.predict([x_test_real,x_test_imag])
            
            # Drawing samples from the saved trained distribution
            samples = self.trainable_distribution.sample((n_samples,),
              bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
            
            ra_samples_x = samples[:,0]
            ra_samples_y = samples[:,1]
            dec_samples = samples[:,2]
            
            # Collecting only valid samples
            ra_samples_x = np.where(ra_samples_x > 1, 1, ra_samples_x)
            ra_samples_x = np.where(ra_samples_x < -1, -1, ra_samples_x)
            
            ra_samples_y = np.where(ra_samples_y > 1, 1, ra_samples_y)
            ra_samples_y = np.where(ra_samples_y < -1, -1, ra_samples_y)

            dec_samples = np.where(dec_samples > np.pi/2, np.pi/2, dec_samples)
            dec_samples = np.where(dec_samples < -np.pi/2, -np.pi/2, dec_samples)
            
            # Converting from (RA_x, RA_y, Dec) to (RA, Dec) co-ordinates 
            ra_samples = np.arctan2(ra_samples_y, ra_samples_x)
            ra_samples = ra_samples + np.pi
            
            eps = 1e-5
            
            # A 2D Kernel Density Estimator is used to find the probability density at ra_pix and de_pix
            zz = self.kde2D(ra_samples,dec_samples, 0.03, ra_pix,de_pix)
            zz = zz/(np.sum(zz) + eps)

            probs.append(zz)
             
            ra_preds.append(ra_samples)
            dec_preds.append(dec_samples)
        
        self.ra_test = self.ra_test + np.pi
        
        # Saving the results
        if(self.n_det == 3):
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyNet/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            f1.create_dataset('H1_SNR', data = self.h1_snr_test)
            f1.create_dataset('L1_SNR', data = self.l1_snr_test)
            f1.create_dataset('V1_SNR', data = self.v1_snr_test)

            f1.close()    
        
        elif(self.n_det == 2):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyNet/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            
            f1.close()
            
            
        
