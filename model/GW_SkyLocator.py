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
        """ Standardize SNR time series data
        Args:
            X_train (float32): Real or imaginary component 
                               of SNR timeseries (training).
            X_test (float32): Real or imaginary component 
                              of SNR timeseries (testing).

        Returns:
            X_train_standardized (float32): Standardized real or imaginary component 
                                            of SNR timeseries (training).
            X_test_standardized (float32): Standardized real or imaginary component 
                                            of SNR timeseries (testing).

        """
        
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        
        X_train_standardized = (X_train - X_train_mean) / X_train_std
        X_test_standardized = (X_test - X_train_mean) / X_train_std
                           
        return X_train_standardized, X_test_standardized
    
    def scale_data(self, data_train:float, data_test:float) -> float:
        """ Standardize scale the labels/intrinsic parameters
        Args:
            data_train (float32): Training set labels/intrinsic parameters.
            data_test (float32): Test set labels/intrinsic parameters.

        Returns:
            data_train_standardized (float32): Standard-scaled training set labels/intrinsic parameters.
            data_test_standardized (float32): Standard-scaled test set labels/intrinsic parameters.
            data_mean (float32): Mean of training set labels/intrinsic parameters.
            data_std (float32): Standard deviation of labels/intrinsic parameters.

        """
        
        data_mean = np.mean(data_train, axis=0)
        data_std = np.std(data_train, axis=0)

        data_train_standardized = (data_train - data_mean) / data_std
        data_test_standardized = (data_test - data_mean) / data_std
            
        return data_train_standardized, data_test_standardized, data_mean, data_std
        
        
    def _preprocess_data(self, d_loader):
        """ Function to pick valid samples: 
            
            1. Atleast 2 det SNR > 3, 
            2. 8 <= Network SNR <= 40)  """
                
        if((self.config.train.PSD == 'design') or (self.config.train.PSD == 'O2') or (self.config.train.PSD == 'O3')):
            
            self.X_train_real, self.X_train_imag, self.y_train, self.ra, self.ra_x, self.ra_y, self.dec, self.intrinsic_train, self.valid, self.net_snr = d_loader.load_valid_samples(self.config, self.X_train_real, self.X_train_imag, self.y_train, self.intrinsic_train, data='train')
            
            self.X_test_real, self.X_test_imag, self.y_test, self.ra_test, self.ra_test_x, self.ra_test_y, self.dec_test, self.intrinsic_test, self.valid_test, self.net_snr_test = d_loader.load_valid_samples(self.config, self.X_test_real, self.X_test_imag, self.y_test, self.intrinsic_test, data='test')
            
        self.gps_time = self.gps_time[self.valid_test]
        
        # Calling standardize function for SNR timeseries data

        self.X_train_real, self.X_test_real = self.standardize_data(self.X_train_real, self.X_test_real)
#        self.X_train_imag, self.X_test_imag = self.standardize_data(self.X_train_imag, self.X_test_imag)
        
        # Reshaping data for 2D CNN
            
        self.X_train = np.hstack((self.X_train_real, self.X_train_imag))
        shape_train = self.X_train.shape[0]
        self.X_train = self.X_train.reshape(shape_train,2,self.config.train.n_samples,self.config.train.num_detectors)
            
        self.X_test = np.hstack((self.X_test_real, self.X_test_imag))
        shape_test = self.X_test.shape[0]
        self.X_test = self.X_test.reshape(shape_test,2,self.config.train.n_samples,self.config.train.num_detectors)
        
        # Scaling intrinsic parameters and labels (RA and Dec).
        
        self.y_train, self.y_test, self.mean, self.std = self.scale_data(self.y_train, self.y_test)
        self.intrinsic_train, self.intrinsic_test, self.intrinsic_mean, self.intrinsic_std = self.scale_data(self.intrinsic_train, self.intrinsic_test)
            
        
        # Converting to float32
        
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
            
            self.input1 = tf.keras.layers.Input([2, self.config.train.n_samples, self.config.train.num_detectors])
            self.input2 = tf.keras.layers.Input(self.intrinsic_train.shape[-1]) 
            self.x_ = tf.keras.layers.Input(shape=self.y_train.shape[-1], dtype=tf.float32)
            
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
            
                self.encoded_features = ResNet34(self.input1, self.input2, self.filters_real, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, input_shapes1=[2, self.config.train.n_samples, self.config.train.num_detectors], input_shapes2 = self.intrinsic_train.shape[-1]).construct_model()
        
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
            self.trainable_distribution = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(loc=np.zeros(3).astype(dtype=np.float32)), bijector = flow_bijector)
        
            return self.trainable_distribution
        
    
    def train(self):
        """Compiles and trains the model"""
        
        with strategy.scope():
            
            flow = self.construct_flow(training=True)
            self.checkpoint = tf.train.Checkpoint(model=flow)
            
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
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=15)
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
            
            # Defining Early Stoppoing callback
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

            # Initializing Callbacks list
            callbacks_list=[custom_checkpoint, early_stopping]  

            # Fitting the model
            model_history = self.model.fit([self.X_train, self.intrinsic_train, self.y_train], np.zeros((len(self.X_train_real)), dtype=np.float32),
              batch_size=self.config.train.batch_size,
              epochs=self.config.train.epochs,
              validation_split=self.config.train.validation_split,
              callbacks=callbacks_list,
              shuffle=True,
              verbose=True)

            self.checkpoint.write(file_prefix=self.checkpoint_prefix)
        
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
                  dtype=np.float32)
    
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
                
     
    def kde2D(self, x:float, y:float, bandwidth:float, ra_pix:float, de_pix:float, **kwargs):
        """Build 2D kernel density estimate (KDE)."""

        xy_sample = np.vstack([de_pix, ra_pix]).T
        xy_train  = np.vstack([y, x]).T

        kde_skl = KernelDensity(kernel='gaussian', bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train) # Fitting the KDE over RA and Dec samples generated by the MAF.

        z = np.exp(kde_skl.score_samples(xy_sample)) # Evaluating probability density over HealPix pixels.
        
#        z = np.exp(self.parrallel_score_samples(kde_skl, xy_sample))
    
        return z, kde_skl

    
    def parrallel_score_samples(self, kde, samples, thread_count=int(multiprocessing.cpu_count())):
        
        with multiprocessing.Pool(thread_count) as p:
            return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))    
    
    
    def _bayestar_adaptive_grid(self, eval_prob_density:bool, top_nside=16, rounds=8):
        
        """Implement of the BAYESTAR adaptive mesh refinement scheme as
            described in Section VI of Singer & Price 2016, PRD, 93, 024013
            :doi:`10.1103/PhysRevD.93.024013`.

            FIXME: Consider refactoring BAYESTAR itself to perform the adaptation
            step in Python.
            """
        probabilities = []
        top_npix = ah.nside_to_npix(top_nside)
        nrefine = top_npix // 4
        cells = zip([0] * nrefine, [top_nside // 2] * nrefine, range(nrefine))
        for iround in range(rounds - 1):
            
#            print('adaptive refinement round {} of {} ...'.format(
#                      iround + 1, rounds - 1))
            cells = sorted(cells, key=lambda p_n_i: p_n_i[0] / p_n_i[1]**2)
            new_nside, new_ipix = np.transpose([
                    (nside * 2, ipix * 4 + i)
                    for _, nside, ipix in cells[-nrefine:] for i in range(4)])
            
            theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
            ra = phi
            dec = 0.5 * np.pi - theta
            
            xy_sample = np.vstack([dec, ra]).T
            
            # If probability density to be directly evaluated using the MAF
            if eval_prob_density:
                
                ra = ra[:,None]
                dec = dec[:,None]
   
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                pixels = np.concatenate([ra_x, ra_y, dec], axis=1)
            
                p = self.flow.prob((pixels),
      bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
        
                p /= tf.reduce_sum(p)
            
                probabilities.append(p)
            
            # If probability density to be evaluated using KDE
            else:
            
    #            p = np.exp(parallel_apply(kde.score_samples, xy_sample, nproc=4))
                p = np.exp(self.kde_obj.score_samples(xy_sample))

    #            p = np.exp(self.parrallel_score_samples(kde, xy_sample))
                probabilities.append(p)
            
            cells[-nrefine:] = zip(p, new_nside, new_ipix)
            
        return cells, probabilities
    
    
    def as_healpix(self, eval_prob_density=False, top_nside=16):
        """Return a HEALPix multi-order map of the posterior density."""
        
        zip_obj, probabilities = self._bayestar_adaptive_grid(eval_prob_density, top_nside=16)
            
        post, nside, ipix = zip(*zip_obj)
        post = np.asarray(list(post))
        nside = np.asarray(list(nside))
        ipix = np.asarray(list(ipix))
        
        post /= np.sum(post * ah.nside_to_pixel_area(nside).to_value(u.sr))

        # Convert from NESTED to UNIQ pixel indices
        order = np.log2(nside).astype(int)
        uniq = moc.nest2uniq(order.astype(np.int8), ipix)

        return Table([uniq, post], names=['UNIQ', 'PROBDENSITY'], copy=False), probabilities

    
    def obtain_samples(self):
        """Obtain samples from trained distribution"""
        
        with tf.device(devices_names[0]):
            
            self.encoder.load_weights('/fred/oz016/Chayan/GW-SkyLocator/model/encoder_models/'+str(self.config.train.network)+'_'+str(self.config.train.dataset)+'_encoder_'+str(self.config.train.num_detectors)+'_det_adaptive_snr-10to20_test.hdf5')
            n_samples = 2000
            probs = []
            ra_preds = []
            dec_preds = []
            probabilities = []
        
            gps_time_GW170817 = 1187008882.4
        
            nside=16
            npix=hp.nside2npix(nside)
            theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))

            # ra_pix and de_pix are co-ordinates in the sky where I want to find the probabilities
            ra_pix = phi
            de_pix = -theta + np.pi/2.0

            for i in range(self.y_test.shape[0]):
            
                x_test = np.expand_dims(self.X_test[i], axis=0)
                intrinsic_test = np.expand_dims(self.intrinsic_test[i], axis=0)
            
                gps_time_test = self.gps_time[i]
            
                self.encoder_features = self.encoder([x_test, intrinsic_test])
                           
                self.flow = self.construct_flow(training=False)
                checkpoint_test = tf.train.Checkpoint(model=self.flow)
                checkpoint_test.restore(self.checkpoint_prefix)
            
                # Generating RA and Dec samples from the learned distribution
                samples = self.flow.sample((n_samples,),
                      bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':self.encoder_features}}))
    
    #            samples = self.trainable_distribution.sample((n_samples,),
    #              bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
                
                # Undoing the scaling of RA and Dec parameters
                samples = self.std*samples + self.mean
    
                ra_samples_x = samples[:,0]
                ra_samples_y = samples[:,1]
                dec_samples = samples[:,2]

    #            ra_samples_x = np.where(ra_samples_x > 1, 1, ra_samples_x)
    #            ra_samples_x = np.where(ra_samples_x < -1, -1, ra_samples_x)
            
    #            ra_samples_y = np.where(ra_samples_y > 1, 1, ra_samples_y)
    #            ra_samples_y = np.where(ra_samples_y < -1, -1, ra_samples_y)

    #            dec_samples = np.where(dec_samples > np.pi/2, np.pi/2, dec_samples)
    #            dec_samples = np.where(dec_samples < -np.pi/2, -np.pi/2, dec_samples)
            
                # Converting cos(RA) and sin(RA) to RA (0 to 2 pi)
                ra_samples = np.arctan2(ra_samples_y, ra_samples_x)
                ra_samples = ra_samples + np.pi
            
                # The model has been trained with injection samples generated with fixed Hanford merger time of GW170817. 
                # During inference, the predicted RA values need to be shifted by an amount equal to the difference 
                # in the GPS times of GW170817 and the test event's merger time to obtain the correct prediction.
                
                detector = Detector('H1')                
                delta = detector.gmst_estimate(gps_time_test) - detector.gmst_estimate(gps_time_GW170817)
                ra_samples = np.mod(ra_samples+delta, 2.0*np.pi)

                eps = 1e-5
            
                # A 2D Kernel Density Estimator is used to find the probability density at ra_pix and de_pix 
                # Fixed resolution skymaps are generated
                
                kde_prob_density, self.kde_obj = self.kde2D(ra_samples,dec_samples, 0.03, ra_pix,de_pix)                 
                kde_prob_density = kde_prob_density/(np.sum(kde_prob_density))    
                probs.append(kde_prob_density)
                       
                # Obtain the multi-order FITS files using adaptive refinement.
                hpmap, probability = self.as_healpix()
                
                # Save FITS files:
            
                # Post-merger injection run 
                io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/BBH/Test_new_BN_test_GPU_'+str(i)+'.fits', hpmap, nest=True)                           

                # Pre-merger injection run
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/45_secs/Test_3_bij_50_epochs_lr_schedule_'+str(i)+'.fits', hpmap, nest=True)
        
                # Real events
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Real_events/Test_3_bij_50_epochs_NSBH_3_det_new_BN_'+str(i)+'.fits', hpmap, nest=True)

                # Bayestar example
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_test/evaluation/skymaps/CPU/Bayestar_test/Test_Bayestar_example_6_'+str(i)+'.fits', hpmap, nest=True)

                # Save generated RA and Dec samples
        
                ra_preds.append(ra_samples)
                dec_preds.append(dec_samples)
                    
            self.ra_test = self.ra_test + np.pi
            
        
        if(self.config.train.num_detectors == 3):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.config.train.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            f1.create_dataset('Valid', data = self.valid_test)
            f1.create_dataset('Network_SNR', data = self.net_snr_test)

            f1.close()    
        
        elif(self.config.train.num_detectors == 2):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.config.train.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            
            f1.close()
                
            
    def obtain_probability_density(self):
        """Obtain probability density from trained distribution"""
        self.encoder.load_weights('/fred/oz016/Chayan/GW-SkyLocator/model/encoder_models/'+str(self.config.train.network)+'_'+str(self.config.train.dataset)+'_encoder_'+str(self.config.train.num_detectors)+'_det_adaptive_snr-10to20_test.hdf5')

        probs = []
        preds_array = []
        probabilities = []
        ra_preds = []
        dec_preds = []
        
        nside=16
        deg2perpix = hp.nside2pixarea(nside)
        npix=hp.nside2npix(nside)
        theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))

        # ra_pix and de_pix are co-ordinates in the sky where I want to find the probabilities
        ra_pix = phi
        de_pix = -theta + np.pi/2.0 
        
        ra_pix = ra_pix - np.pi
        ra_pix_x = np.cos(ra_pix)
        ra_pix_y = np.sin(ra_pix)
         
        ra_pix = ra_pix[:,None]
        ra_pix_x = ra_pix_x[:,None]
        ra_pix_y = ra_pix_y[:,None]
        de_pix = de_pix[:,None]

        pixels = np.concatenate([ra_pix_x, ra_pix_y, de_pix], axis=1)
    
        eps = 1e-5
       

        for i in range(self.y_test.shape[0]):
            
            x_test = np.expand_dims(self.X_test[i], axis=0)
            intrinsic_test = np.expand_dims(self.intrinsic_test[i], axis=0)
            
    #            start = timeit.default_timer()
    
            self.encoder_features = self.encoder([x_test, intrinsic_test])

            self.flow = self.construct_flow(training=False)
            checkpoint_test = tf.train.Checkpoint(model=self.flow)
            checkpoint_test.restore(self.checkpoint_prefix)
                       
            # Obtain the probability density directly using the MAF.
            prob_density = tf.math.exp(self.flow.log_prob(pixels, bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}})))
                        

            # Obtain the multi-order FITS files using adaptive refinement.       
            hpmap, probability = self.as_healpix(eval_prob_density=True)
            
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/NSBH/skymaps/Gaussian_KDE/Test_3_bij_lr_schedule_'+str(i)+'.fits', hpmap, nest=True)

    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/15_secs/Test_3_bij_50_epochs_prob_density_'+str(i)+'.fits', hpmap, nest=True)
            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/Real_events/Test_3_bij_50_epochs_BNS_3_det_prob_density_'+str(i)+'.fits', hpmap, nest=True)
    
            probabilities.append(probability)
                 
                        
        self.ra_test = self.ra_test + np.pi
        
        if(self.config.train.num_detectors == 3):
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.config.train.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('Probabilities_adaptive', data = probabilities)
            f1.create_dataset('Preds', data = preds_array)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            f1.create_dataset('Valid', data = self.valid_test)

            f1.close()    
        
        elif(self.config.train.num_detectors == 2):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.config.train.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            
            f1.close()
                    
