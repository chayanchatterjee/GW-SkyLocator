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
from utils.custom_checkpoint import CustomCheckpoint 

# external
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np
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


class GW_SkyNet(BaseModel):
    """Normalizing Flow Model Class"""
    def __init__(self, config):
        
        super().__init__(config)
        
        self.network = self.config.train.network
        self.X_train_real = []
        self.X_train_imag = []
        self.X_test_real = []
        self.X_test_imag = []
        self.y_train = []
        self.y_test = []
        self.ra = []
        self.dec = []
        self.ra_test = []
        self.dec_test = []

        self.encoded_features = None
        self.model = None
        self.encoder = None
        self.sc = None
        
        self.num_train = self.config.train.num_train
        self.num_test = self.config.train.num_test
        self.n_samples = self.config.train.n_samples
        self.n_det = self.config.train.num_detectors
        self.epochs = self.config.train.epochs
        self.lr = self.config.model.learning_rate
        self.batch_size = self.config.train.batch_size
        self.val_split = self.config.train.validation_split
        
        if(self.network == 'WaveNet'):
            self.filters = self.config.model.WaveNet.filters
            self.kernel_size = self.config.model.WaveNet.kernel_size
            self.activation = self.config.model.WaveNet.activation
            self.dilation_rate = self.config.model.WaveNet.dilation_rate
            
        elif(self.network == 'ResNet'):
            self.kernels_res = self.config.model.ResNet.kernels_resnet_block
            self.stride_res = self.config.model.ResNet.stride_resnet_block
            self.kernel_size_res = self.config.model.ResNet.kernel_size_resnet_block
            self.kernels = self.config.model.ResNet.kernel_size
            self.kernel_size = self.config.model.ResNet.kernel_size
            self.strides = self.config.model.ResNet.strides
            
        elif(self.network == 'ResNet-34'):
            self.filters = self.config.model.ResNet_34.filters
            self.kernel_size = self.config.model.ResNet_34.kernel_size
            self.strides = self.config.model.ResNet_34.strides
            self.pool_size = self.config.model.ResNet_34.pool_size
            self.prev_filters = self.config.model.ResNet_34.prev_filters
            
        self.num_bijectors = self.config.model.num_bijectors
        self.trainable_distribution = None
        self.MAF_hidden_units = self.config.model.MAF_hidden_units
        
    def load_data(self):
        """Loads and Preprocess data """
        
        self.X_train_real, self.X_train_imag = DataLoader().load_train_data(self.config.data)
        self.X_test_real, self.X_test_imag = DataLoader().load_test_data(self.config.data, self.num_test, self.n_samples)
        
        self.y_train = DataLoader().load_train_parameters(self.config.parameters)
        self.y_test, self.ra_test, self.dec_test = DataLoader().load_test_parameters(self.config.parameters)
        
        self._preprocess_data()
        
    def _preprocess_data(self):
        """ Removing < 3 det samples and scaling RA and Dec values """
        
        self.X_train_real, self.X_train_imag, self.y_train, self.ra, self.dec = DataLoader().load_3_det_samples(self.config.parameters, self.X_train_real, self.X_train_imag, self.y_train, self.num_train, data='train')
        
        self.X_test_real, self.X_test_imag, self.y_test, self.ra_test, self.dec_test = DataLoader().load_3_det_samples(self.config.parameters, self.X_test_real, self.X_test_imag, self.y_test, self.num_test, data='test')
                
        self.sc = StandardScaler()
        self.y_train = self.sc.fit_transform(self.y_train)
        self.y_test = self.sc.transform(self.y_test)
        
        self.X_train_real = self.X_train_real.astype("float32")
        self.X_train_imag = self.X_train_imag.astype("float32")
        self.y_train = self.y_train.astype("float32")

        self.X_test_real = self.X_test_real.astype("float32")
        self.X_test_imag = self.X_test_imag.astype("float32")
        self.y_test = self.y_test.astype("float32")
        
    def construct_model(self):
        """ Constructing the neural network encoder model
        
        Args:
            model_type:     'wavenet', 'resnet', 'resnet-34'
            
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
                
            
            x_ = tf.keras.layers.Input(shape=self.y_train.shape[-1], dtype=tf.float32)
        
            # Define a more expressive model
            bijectors = []

            for i in range(self.num_bijectors):
                masked_auto_i = self.make_masked_autoregressive_flow(i, hidden_units = self.MAF_hidden_units, activation = 'relu',
                                            conditional_event_shape=self.encoded_features.shape[-1])
                
                bijectors.append(masked_auto_i)
    
#                USE_BATCHNORM = True
    
#                if USE_BATCHNORM and i % 2 == 0:
                # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
#                    bijectors.append(tfb.BatchNormalization(name='batch_normalization'+str(i)))

                bijetors.append(tfb.BatchNormalization(name='batch_normalization'+str(i)))
    
                bijectors.append(tfb.Permute(permutation = [1, 0]))
                flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
            
            # Define the trainable distribution
            self.trainable_distribution = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(2)),
                        bijector = flow_bijector)

            log_prob_ = self.trainable_distribution.log_prob(x_, bijector_kwargs=
                        self.make_bijector_kwargs(self.trainable_distribution.bijector, 
                                             {'maf.': {'conditional_input':self.encoded_features}}))

            self.model = tf.keras.Model([input1, input2, x_], log_prob_)
            self.encoder = tf.keras.Model([input1, input2], self.encoded_features)  
  
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)  # optimizer
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=self.model)
        
        self.train(log_prob_, checkpoint)
    

    # Define the trainable distribution
    def make_masked_autoregressive_flow(self, index, hidden_units, activation, conditional_event_shape):
    
        made = tfp.bijectors.AutoregressiveNetwork(params=2,
              hidden_units=hidden_units,
              event_shape=(2,),
              activation=activation,
              conditional=True,
              kernel_initializer=tf.keras.initializers.VarianceScaling(0.1),
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
        
        custom_checkpoint = CustomCheckpoint(filepath='model/'+str(self.network)+'_encoder_3_det.hdf5',encoder=self.encoder)
        
        self.model.compile(optimizer=tf.optimizers.Adam(lr=self.lr), loss=lambda _, log_prob: -log_prob)
        self.model.summary()
                                             
        # initialize checkpoints
        dataset_name = "checkpoints/NSBH_3_det"
        checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

        callbacks_list=[custom_checkpoint]  

        self.model.fit([self.X_train_real, self.X_train_imag, self.y_train], np.zeros((len(self.X_train_real), 0), dtype=np.float32),
              batch_size=self.batch_size,
              epochs=self.epochs,
              validation_split=self.val_split,
              callbacks=callbacks_list,
              shuffle=True,
              verbose=True)

        checkpoint.save(file_prefix=checkpoint_prefix)
        
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

        
     
    def obtain_samples(self):
        """Obtain samples from trained distribution"""
        
        self.encoder.load_weights("model/"+str(self.network)+"_encoder_3_det.hdf5")
        
        n_samples = 5000
        probs = []
        
        nside=32
        npix=hp.nside2npix(nside)
        theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))

        # ra_pix and de_pix are co-ordinates in the skuy where I want to find the probabilities
        ra_pix = phi
        de_pix = -theta + np.pi/2.0 

        for i in range(self.y_test.shape[0]):
            x_test_real = np.expand_dims(self.X_test_real[i],axis=0)
            x_test_imag = np.expand_dims(self.X_test_imag[i],axis=0)
    
            preds = self.encoder.predict([x_test_real,x_test_imag])
    
            samples = self.trainable_distribution.sample((n_samples,),
              bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
            
            samples = self.sc.inverse_transform(samples)
#            self.y_test = self.sc.inverse_transform(self.y_test)
    
            ra_samples = samples[:,0]
            dec_samples = samples[:,1]
            
            # A 2D Kernel Density Estimator is used to find the probability density at ra_pix and de_pix
            zz = self.kde2D(ra_samples,dec_samples, 0.03, ra_pix,de_pix)
            zz = zz/(np.sum(zz))

            probs.append(zz)
    
        f1 = h5py.File('evaluation/Injection_run_SNR_time_series_NSBH_NF_3_det_new_model_ResNet.hdf', 'w')
        f1.create_dataset('Probabilities', data = probs)
        f1.create_dataset('RA_test', data = self.ra_test)
        f1.create_dataset('Dec_test', data = self.dec_test)

        f1.close()    
        
        
            
            
            
    
        
        
                                             
                                             
                                             
    
    
    

                
                
          
        
            
            
            
        
        
            
    




