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
from .resnet_50 import ResNet50
from .resnet_34_2_det import ResNet_34_2_det
from .lstm import LSTM_model
from .cnn import CNN_model
from utils.custom_checkpoint import CustomCheckpoint 
from utils.simulated_annealing import SimulatedAnnealingCallback
from utils.halt_callback import haltCallback
from utils.cosine_annealing import SGDRScheduler
from utils.batch_normalization import BatchNormalization
from utils.mp import parallel_apply

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
import timeit

import healpy as hp
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter
from numba import jit, cuda
import multiprocessing

import ligo.skymap.kde as KDE
from ligo.skymap import io
import astropy_healpix as ah
from ligo.skymap.kde import moc
from astropy.table import Table
from astropy import units as u
from pycbc.detector import Detector
tf.keras.backend.set_floatx('float64')

# GPU specifications

device_type = 'GPU'
n_gpus = 2
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])


class GW_SkyLocator(BaseModel):
    """Normalizing Flow Model Class"""
    def __init__(self, config):
        
        super().__init__(config)
        
        self.encoded_features = None
        self.model = None
        self.encoder = None
        
        if(self.config.train.network == 'WaveNet'):
            self.filters = self.config.model.WaveNet.filters
            self.kernel_size = self.config.model.WaveNet.kernel_size
            self.activation = self.config.model.WaveNet.activation
            self.dilation_rate = self.config.model.WaveNet.dilation_rate
            
        elif(self.config.train.network == 'ResNet'):
            self.kernels_res = self.config.model.ResNet.kernels_resnet_block
            self.stride_res = self.config.model.ResNet.stride_resnet_block
            self.kernel_size_res = self.config.model.ResNet.kernel_size_resnet_block
            self.kernels = self.config.model.ResNet.kernels
            self.kernel_size = self.config.model.ResNet.kernel_size
            self.strides = self.config.model.ResNet.strides
            
        elif(self.config.train.network == 'ResNet-34'):
            self.filters_real = self.config.model.ResNet_34.filters_real
            self.filters_imag = self.config.model.ResNet_34.filters_imag
            self.kernel_size = self.config.model.ResNet_34.kernel_size
            self.strides = self.config.model.ResNet_34.strides
            self.pool_size = self.config.model.ResNet_34.pool_size
            self.prev_filters_real = self.config.model.ResNet_34.prev_filters_real
            self.prev_filters_imag = self.config.model.ResNet_34.prev_filters_imag
            
        elif(self.config.train.network == 'ResNet-50'):
            self.filters_real = self.config.model.ResNet_50.filters_real
            self.filters_imag = self.config.model.ResNet_50.filters_imag
            self.kernel_size = self.config.model.ResNet_50.kernel_size
            self.strides = self.config.model.ResNet_50.strides
            self.pool_size = self.config.model.ResNet_50.pool_size
            self.prev_filters_real = self.config.model.ResNet_50.prev_filters_real
            self.prev_filters_imag = self.config.model.ResNet_50.prev_filters_imag
            
        elif(self.config.train.network == 'ResNet-34_2_det'):
            self.filters_real = self.config.model.ResNet_34_2_det.filters_real
            self.filters_imag = self.config.model.ResNet_34_2_det.filters_imag
            self.kernel_size = self.config.model.ResNet_34_2_det.kernel_size
            self.strides = self.config.model.ResNet_34_2_det.strides
            self.pool_size = self.config.model.ResNet_34_2_det.pool_size
            self.prev_filters_real = self.config.model.ResNet_34_2_det.prev_filters_real
            self.prev_filters_imag = self.config.model.ResNet_34_2_det.prev_filters_imag
            
        elif(self.config.train.network == 'LSTM'):
            self.n_units = self.config.model.LSTM_model.n_units
            self.rate = self.config.model.LSTM_model.rate
            
        elif(self.config.train.network == 'CNN'):
            self.filters = self.config.model.CNN_model.filters
            self.kernel_size = self.config.model.CNN_model.kernel_size
            self.max_pool_size = self.config.model.CNN_model.max_pool_size
            self.rate = self.config.model.CNN_model.dropout_rate
            self.n_units = self.config.model.CNN_model.n_units

        self.num_bijectors = self.config.model.num_bijectors
        self.trainable_distribution = None
        self.MAF_hidden_units = self.config.model.MAF_hidden_units
        
    def load_data(self):
        """Loads SNR timeseries data, intrinsic parameters and labels (RA and Dec) """
        
        d_loader = DataLoader()
        
        if(self.config.train.num_detectors == 3):
            
            self.X_train_real, self.X_train_imag = d_loader.load_train_3_det_data(self.config)
            self.X_test_real, self.X_test_imag = d_loader.load_test_3_det_data(self.config)
                
            self.y_train, self.intrinsic_train = d_loader.load_train_3_det_parameters(self.config)
            
            self.y_test, self.ra_test, self.dec_test, self.gps_time, self.intrinsic_test = d_loader.load_test_3_det_parameters(self.config)
            
        elif(self.config.train.num_detectors == 2):
            
            self.X_train_real, self.X_train_imag = d_loader.load_train_2_det_data(self.config)
            self.X_test_real, self.X_test_imag = d_loader.load_test_2_det_data(self.config)
                
            self.y_train, self.intrinsic_train = d_loader.load_train_2_det_parameters(self.config)
            
            self.y_test, self.ra_test, self.dec_test, self.gps_time, self.intrinsic_test = d_loader.load_test_2_det_parameters(self.config)
        
        self._preprocess_data(d_loader)
        
        
    def standardize_data(self, X_train:float, X_test:float) -> float:
        """ Standardize SNR time series data sample by sample
        Args:
            X_train (float32): Real component of SNR timeseries (training).
            X_test (float32): Real component of SNR timeseries (testing).

        Returns:
            X_train_standardized (float32): Standardized real component 
                                            of SNR timeseries (training).
            X_test_standardized (float32): Standardized real component 
                                            of SNR timeseries (testing).

        """
        
        X_train_real_std = np.std(X_train, axis=1)
        X_test_real_std = np.std(X_test, axis=1)
        
        X_train_standardized = X_train/ X_train_real_std[:,None,:]
        X_test_standardized = X_test/ X_test_real_std[:,None,:]
    
        return X_train_standardized, X_test_standardized
    
    def scale_data(self, data_train:float, data_test:float) -> float:
        """ Standardize scale the labels/intrinsic parameters
        Args:
            data_train (float64): Training set labels/intrinsic parameters.
            data_test (float64): Test set labels/intrinsic parameters.

        Returns:
            data_train_standardized (float64): Standard-scaled training set labels/intrinsic parameters.
            data_test_standardized (float64): Standard-scaled test set labels/intrinsic parameters.
            data_mean (float64): Mean of training set labels/intrinsic parameters.
            data_std (float64): Standard deviation of labels/intrinsic parameters.

        """
        
        data_mean = np.mean(data_train, axis=0)
        data_std = np.std(data_train, axis=0)

        data_train_standardized = (data_train - data_mean) / data_std
        data_test_standardized = (data_test - data_mean) / data_std
            
        return data_train_standardized, data_test_standardized, data_mean, data_std
        
        
    def _preprocess_data(self, d_loader):
        """ Function to pick valid samples: 
            
            1. Atleast 2 det SNR > 3, 
            2. 8 <= Network SNR <= 40  """
                
        if((self.config.train.PSD == 'design') or (self.config.train.PSD == 'O2') or (self.config.train.PSD == 'O3')):
            
            self.X_train_real, self.X_train_imag, self.y_train, self.ra, self.ra_x, self.ra_y, self.dec, self.intrinsic_train, self.valid, self.net_snr = d_loader.load_valid_samples(self.config, self.X_train_real, self.X_train_imag, self.y_train, self.intrinsic_train, data='train')
            
            self.X_test_real, self.X_test_imag, self.y_test, self.ra_test, self.ra_test_x, self.ra_test_y, self.dec_test, self.intrinsic_test, self.valid_test, self.net_snr_test = d_loader.load_valid_samples(self.config, self.X_test_real, self.X_test_imag, self.y_test, self.intrinsic_test, data='test')
            
        self.gps_time = self.gps_time[self.valid_test]
        
        # Calling standardize function for SNR timeseries data

        self.X_train_real, self.X_test_real = self.standardize_data(self.X_train_real, self.X_test_real)
#        self.X_train_imag, self.X_test_imag = self.standardize_data(self.X_train_imag, self.X_test_imag)
        
        # Reshaping data for 1D CNN
            
        self.X_train = np.hstack((self.X_train_real, self.X_train_imag))      
        self.X_test = np.hstack((self.X_test_real, self.X_test_imag))
      
        # Scaling intrinsic parameters and labels (RA and Dec).
        
#        self.y_train, self.y_test, self.mean, self.std = self.scale_data(self.y_train, self.y_test)
        self.intrinsic_train, self.intrinsic_test, self.intrinsic_mean, self.intrinsic_std = self.scale_data(self.intrinsic_train, self.intrinsic_test)
            
        
        # Converting to float64
        
        self.X_train_real = self.X_train_real.astype("float64")
        self.X_train_imag = self.X_train_imag.astype("float64")
        self.y_train = self.y_train.astype("float64")

        self.X_test_real = self.X_test_real.astype("float64")
        self.X_test_imag = self.X_test_imag.astype("float64")
        self.y_test = self.y_test.astype("float64")
        
    def construct_model(self):
        """ Constructing the neural network encoder model
        
        Args:
            model_type:     'wavenet', 'resnet', 'resnet-34'
            
            kwargs:         Based on the model_type
            
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
               
               'resnet-34'  input_dim_real   [2, n_samples, n_detectors]
                            input_dim_imag   [2, n_samples, n_detectors]
                            filters          Number of filters in main layer
                            kernel_size      Kernel size in main layer
                            strides          Strides in main layer
                            prev_filters     Number of filters in previous main/Residual layer
                            input_shapes     Shapes of input signals
                            
               'resnet-50'  input_dim_real   [n_samples, n_detectors]
                            input_dim_imag   [n_samples, n_detectors]
                            kernels_res      Number of kernels in ResNet block
                            stride_res       Stride in ResNet block
                            kernel_size_res  Kernel size in ResNet block
                            kernels          Number of kernels in CNN layers
                            kernel_size      Kernel size in CNN layers
                            strides          Strides in CNN layers
                         
        """
        with strategy.scope(): 
            
            self.input1 = tf.keras.layers.Input([self.config.train.n_samples, self.config.train.num_detectors])
            self.input2 = tf.keras.layers.Input(self.intrinsic_train.shape[-1]) 
            self.x_ = tf.keras.layers.Input(shape=self.y_train.shape[-1], dtype=tf.float64)
            
            if(self.config.train.network == 'WaveNet'):
            
                filters = self.filters
                kernel_size = self.kernel_size
                activation = self.activation
                dilation_rate = self.dilation_rate
                
                self.encoded_features = WaveNet(input1, input2, self.filters, self.kernel_size, self.activation, self.dilation_rate).construct_model()
            
            elif(self.config.train.network == 'ResNet'):
            
                self.encoded_features = ResNet(input1, input2, self.kernels_res, self.kernel_size_res, self.stride_res,
                                               self.kernels, self.kernel_size, self.strides).construct_model()
            elif(self.config.train.network == 'ResNet-34'):
            
                self.encoded_features = ResNet34(self.input1, self.input2, self.filters_real, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, input_shapes1=[2*self.config.train.n_samples, self.config.train.num_detectors], input_shapes2 = self.intrinsic_train.shape[-1]).construct_model()
        
            elif(self.config.train.network == 'ResNet-50'):
                               
                self.encoded_features = ResNet50(input1, input2, self.filters_real, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, input_shapes1=[self.config.train.n_samples, self.config.train.num_detectors], input_shapes2 = self.intrinsic_train.shape[-1]).construct_model()
    
    
    def construct_flow(self, training:bool):
        """ Constructing the Masked Autoregressive Flow model
        
        Args (from config.json):
        
            num_bijectors (int):  Number of MAF blocks
            MAF_hidden_units (list of integers): Number of neurons in hidden layers of MAF
            
        Returns:
            trainable_distribution (TensorFlow object): The approximate target distribution learnt by the MAF.
                         
        """
               
        
        with strategy.scope():
            # Define a more expressive model
            bijectors = []
    #            bijectors.append(tfb.BatchNormalization(training=training, name='batch_normalization'))
            bijectors.append(BatchNormalization())
        
            for i in range(self.num_bijectors):            
                
    #                bijectors.append(tfb.BatchNormalization(name='batch_normalization'+str(i)))
                masked_auto_i = self.make_masked_autoregressive_flow(i, hidden_units = self.MAF_hidden_units, activation = 'relu', conditional_event_shape=self.encoded_features.shape[-1])
                
                bijectors.append(masked_auto_i)
                bijectors.append(tfb.Permute(permutation = [2, 1, 0]))
    
                USE_BATCHNORM = True
    
    #                if (i+1) % int(2) == 0:

                if USE_BATCHNORM:
               
               # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
    #                    bijectors.append(tfb.BatchNormalization(training=training,name='batch_normalization'+str(i)))
                    bijectors.append(BatchNormalization())

        #                bijectors.append(tfb.BatchNormalization(name='batch_normalization'+str(i)))
            
            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
            
                # Define the trainable distribution
            self.trainable_distribution = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(loc=np.zeros(3).astype(dtype=np.float64)), bijector = flow_bijector)
        
            return self.trainable_distribution
        
    
    def train(self):
        """Compiles and trains the model"""
        
        with strategy.scope():
            
            flow = self.construct_flow(training=True)
            ckpt = tf.train.Checkpoint(model=flow)
            
            # Setting up the loss function 
            log_prob_ = -tf.reduce_mean(flow.log_prob(self.x_, bijector_kwargs=
                        self.make_bijector_kwargs(self.trainable_distribution.bijector, 
                                             {'maf.': {'conditional_input':self.encoded_features}})))

            self.model = tf.keras.Model([self.input1, self.input2, self.x_], log_prob_)
            self.encoder = tf.keras.Model([self.input1, self.input2], self.encoded_features) 

            # Settings for Polynoimal decay learning rate
            base_lr = 2e-4
            end_lr = 5e-5
            max_epochs = self.config.train.epochs  # maximum number of epochs of the training
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

            opt = tf.keras.optimizers.Adam(learning_rate=self.config.model.learning_rate)  # optimizer
            
            # Initialize checkpoints for MAF and encoder networks
            dataset_name = "/fred/oz016/Chayan/GW-SkyLocator/checkpoints/"+str(self.config.train.dataset)+"_"+str(self.config.train.num_detectors)+"_det_"+str(self.config.train.network)+"_adaptive"
            checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
            self.checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
            
            custom_checkpoint = CustomCheckpoint(filepath='/fred/oz016/Chayan/GW-SkyLocator/model/encoder_models/'+str(self.config.train.network)+'_'+str(self.config.train.dataset)+'_encoder_'+str(self.config.train.num_detectors)+'_det_adaptive_snr-10to20_test.hdf5',encoder=self.encoder)
            
        
            # Load best model with min validation loss
            if(self.config.train.checkpoint_restore):
                
                self.checkpoint.restore(self.checkpoint_prefix)
                self.encoder.load_weights("/fred/oz016/Chayan/GW-SkyNet_pre-merger/model/encoder_models/"+str(self.config.train.network)+"_"+str(self.config.train.dataset)+"_encoder_"+str(self.config.train.num_detectors)+"_det_adaptive_snr-10to20_test.hdf5")

        
            trainingStopCallback = haltCallback(self.model)
            
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.model.learning_rate), loss=lambda _, log_prob: log_prob)
    
            self.model.summary()
                                             
            # Settings for Reduce Learning Rate callback
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=30)
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
            
            # Defining Early Stoppoing callback
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

            # Initializing Callbacks list
            callbacks_list=[custom_checkpoint, reduce_lr]  

            # Fitting the model
            model_history = self.model.fit([self.X_train, self.intrinsic_train, self.y_train], np.zeros((len(self.X_train_real)), dtype=np.float32),
              batch_size=self.config.train.batch_size,
              epochs=self.config.train.epochs,
              validation_split=self.config.train.validation_split,
              callbacks=callbacks_list,
              shuffle=True,
              verbose=True)

            ckpt.write(self.checkpoint_prefix)
          
            self.encoder.save_weights('/fred/oz016/Chayan/GW-SkyNet_pre-merger/model/encoder_models/'+str(self.network)+'_'+str(self.dataset)+'_encoder_'+str(self.n_det)+'_det_adaptive_snr-10to20_test')
          
            self.plot_loss_curves(model_history.history['loss'], model_history.history['val_loss'])        
        

        # Define the trainable distribution
    def make_masked_autoregressive_flow(self, index:int, hidden_units, activation:str, conditional_event_shape):
        """ Setting up the MADE block of MAF model
         
        Args:        
            index (int):  Index of bijector.
            hidden_units (list of integers): Number of neurons in hidden layers of MADE.
            activation (string): Activation function of MADE.
            conditional_event_shape (tuple): Shape of conditional input to MAF model.
            
        Returns:
            MAF bijector (TensorFlow object): The MAF bijector that transform the base distribution
                                              to the target distribution.
                                
        """
        
        made = tfp.bijectors.AutoregressiveNetwork(params=2,
                  hidden_units=hidden_units,
                  event_shape=(3,),
                  activation=activation,
                  conditional=True,
                  kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1),
                  conditional_event_shape=conditional_event_shape,
                  dtype=np.float64)
    
        return tfp.bijectors.MaskedAutoregressiveFlow(shift_and_log_scale_fn = made, name='maf'+str(index))

    def make_bijector_kwargs(self, bijector, name_to_kwargs):
        """ Setting up kwargs for conditional input of MAF """
        
        if hasattr(bijector, 'bijectors'):
            
            return {b.name: self.make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    
        else:
            
            for name_regex, kwargs in name_to_kwargs.items():
                
                if re.match(name_regex, bijector.name):
                    
                    return kwargs
        
        return {}
    
    def scheduler(self, epochs:int, lr:float) -> float:
        """ Function for Learning Rate Scheduler callback
        
        Args: epochs (int): Current epoch number.
              lr (float): Current learning rate.
              
        Returns: Updated learnig rate.
        
        """
        
        if epochs < 50:
            return lr
        elif epochs >= 50:
            return lr * tf.math.exp(-0.1)
   
    def plot_loss_curves(self, loss, val_loss):
        """ Plots training and validation loss curves """
        # summarize history for accuracy and loss
        plt.figure(figsize=(6, 4))
        plt.plot(loss, "r--", label="Loss on training data")
        plt.plot(val_loss, "r", label="Loss on validation data")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
            
        plt.savefig("/fred/oz016/Chayan/GW-SkyLocator/evaluation/Loss_curves/Accuracy_curve_"+str(self.config.train.network)+"_"+str(self.config.train.dataset)+"_"+str(self.config.train.num_detectors)+"_det_test.png", dpi=200)
                
    def bayestar_adaptive_grid(self, probdensity, flow, preds, delta, *args, top_nside=16, rounds=8,
                           **kwargs):
        """Create a sky map by evaluating a function on an adaptive grid.

        Perform the BAYESTAR adaptive mesh refinement scheme as described in
        Section VI of Singer & Price 2016, PRD, 93, 024013
        :doi:`10.1103/PhysRevD.93.024013`. This computes the sky map
        using a provided analytic function and refines the grid, dividing the
        highest 25% into subpixels and then recalculating their values. The extra
        given args and kwargs will be passed to the given probdensity function.

        Parameters
        ----------
        probdensity : callable
            Probability density function. The first argument consists of
            column-stacked array of right ascension and declination in radians.
            The return value must be a 1D array of the probability density in
            inverse steradians with the same length as the argument.
        top_nside : int
            HEALPix NSIDE resolution of initial evaluation of the sky map
        rounds : int
            Number of refinement rounds, including the initial sky map evaluation

        Returns
        -------
        skymap : astropy.table.Table
            An astropy Table with UNIQ and PROBDENSITY columns, representing
            a multi-ordered sky map
        """
        probs = []
        top_npix = ah.nside_to_npix(top_nside)
        nrefine = top_npix // 4
        cells = zip([0] * nrefine, [top_nside // 2] * nrefine, range(nrefine))
        for iround in range(rounds - 1):
            
            print('adaptive refinement round {} of {} ...'.format(
                iround+1, rounds-1))
            cells = sorted(cells, key=lambda p_n_i: p_n_i[0] / p_n_i[1]**2)
            new_nside, new_ipix = np.transpose([
                (nside * 2, ipix * 4 + i)
                for _, nside, ipix in cells[-nrefine:] for i in range(4)])
          
            theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
            
            ra = phi
            ra = np.mod(ra-delta, 2.0*np.pi)
            
            dec = 0.5 * np.pi - theta  
            
            dec = dec
                
            ra = ra[:,None]
            dec = dec[:,None]
    
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            pixels = np.concatenate([ra_x, ra_y, dec], axis=1)
                        
            p = probdensity(flow,pixels,preds)            
            
            probs.append(p)
            
            cells[-nrefine:] = zip(p, new_nside, new_ipix)
    
        """Return a HEALPix multi-order map of the posterior density."""
        post, nside, ipix = zip(*cells)
        post = np.asarray(list(post))
        nside = np.asarray(list(nside))
        ipix = np.asarray(list(ipix))
    
        # Make sure that sky map is normalized (it should be already)
        post /= np.sum(post * ah.nside_to_pixel_area(nside).to_value(u.sr))

        # Convert from NESTED to UNIQ pixel indices
        order = np.log2(nside).astype(int)
        uniq = nest2uniq(order.astype(np.int8), ipix)

        # Done!
        return Table([uniq, post], names=['UNIQ', 'PROBDENSITY'], copy=False), probs
                             
    
    def nf_prob_density(self,flow,pixels,preds):
        
        return flow.prob((pixels),
                bijector_kwargs=self.make_bijector_kwargs(flow.bijector, {'maf.': {'conditional_input':preds}}))
    
#        return np.exp(self.model.predict([x_test, intrinsic_test, pixels]))

    
    
    def obtain_probability_density(self):
        """Obtain probability density from trained distribution"""
        
        self.encoder.load_weights('/fred/oz016/Chayan/GW-SkyNet_pre-merger/model/encoder_models/'+str(self.network)+'_'+str(self.dataset)+'_encoder_'+str(self.n_det)+'_det_adaptive_snr-10to20_test')

        flow = self.construct_flow(training=False)
        checkpoint = tf.train.Checkpoint(flow)
        checkpoint.restore(self.checkpoint_prefix)
            
        
        probs = []
        probs_nf = []
        preds_array = []
        probabilities = []
        ra_preds = []
        dec_preds = []

        eps = 1e-5

        for i in range(self.y_test.shape[0]):
            
            x_test = np.expand_dims(self.X_test[i], axis=0)
            intrinsic_test = np.expand_dims(self.intrinsic_test[i], axis=0)
            
            gps_time_test = self.gps_time[i]
            
            nside=16
            deg2perpix = hp.nside2pixarea(nside)
            npix=hp.nside2npix(nside)
            theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))

            # ra_pix and de_pix are co-ordinates in the skuy where I want to find the probabilities
            ra_pix = phi
            de_pix = -theta + np.pi/2.0 
    
            gps_time_GW170817 = 1187008882.4
            
        
            detector = Detector('H1')
            delta = detector.gmst_estimate(gps_time_test) - detector.gmst_estimate(gps_time_GW170817)

            ra_pix = np.mod(ra_pix-delta, 2.0*np.pi)

        
            ra_pix = ra_pix - np.pi
            ra_pix_x = np.cos(ra_pix)
            ra_pix_y = np.sin(ra_pix)
         
            ra_pix = ra_pix[:,None]
            ra_pix_x = ra_pix_x[:,None]
            ra_pix_y = ra_pix_y[:,None]
            de_pix = de_pix[:,None]

            pixels = np.concatenate([ra_pix_x, ra_pix_y, de_pix], axis=1)
        
            preds = self.encoder([x_test, intrinsic_test])
            preds_array.append(preds)
        
            prob_density = flow.prob((pixels), bijector_kwargs=self.make_bijector_kwargs(flow.bijector, {'maf.': {'conditional_input':preds}}))

            prob_density = np.array(prob_density)
            probs.append(prob_density)
                                         

            # Uncomment for everything except 0 secs     
            
            hpmap, probs_nf_sample = self.bayestar_adaptive_grid(self.nf_prob_density, flow, preds, delta)
            probs_nf.append(probs_nf_sample)
        
                
            ra_preds.append(ra_samples)
            dec_preds.append(dec_samples)
            
                       
            # Test with Bayestar SNR time series
            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_test/evaluation/skymaps/CPU/Bayestar_test/Test_Bayestar_coinc_BNS_prob_density_'+str(i)+'.fits', hpmap, nest=True)
   

        probs_nf = np.array(probs_nf)
        ra_preds = np.array(ra_preds)
        dec_preds = np.array(dec_preds)
                                                 
        if(self.n_det == 3):
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('Probabilities_NF', data = probs_nf)
            f1.create_dataset('RA_samples', data= ra_preds)
            f1.create_dataset('Dec_samples', data= dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            f1.create_dataset('Valid', data = self.valid_test)

            f1.close()    
        
        elif(self.n_det == 2):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
    #            f1.create_dataset('RA_samples', data = ra_preds)
    #            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            
            f1.close()
                    
 
    
