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
#try:
#    tf.config.experimental.set_memory_growth(devices[1], True)
#except:
  # Invalid device or cannot modify virtual devices once initialized.
#    pass


class GW_Locator(BaseModel):
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
        self.valid = []
        self.valid_test = []

        self.encoded_features = None
        self.model = None
        self.encoder = None
        self.sc = None
        
        self.num_train = self.config.train.num_train
        self.num_test = self.config.train.num_test
        self.n_samples = self.config.train.n_samples
        self.train_real = self.config.train.train_real
        self.test_real = self.config.train.test_real
        self.psd = self.config.train.PSD
        self.train_negative_latency = self.config.train.train_negative_latency
        self.train_negative_latency_seconds = self.config.train.train_negative_latency_seconds
        self.test_negative_latency = self.config.train.test_negative_latency
        self.test_negative_latency_seconds = self.config.train.test_negative_latency_seconds
        self.snr_range_train = self.config.train.snr_range_train
        self.snr_range_test = self.config.train.snr_range_test
        self.min_snr = self.config.train.min_snr
        self.n_det = self.config.train.num_detectors
        self.epochs = self.config.train.epochs
        self.lr = self.config.model.learning_rate
        self.batch_size = self.config.train.batch_size
        self.val_split = self.config.train.validation_split
        self.output_filename = self.config.train.output_filename
        
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
            self.filters_real = self.config.model.ResNet_34.filters_real
            self.filters_imag = self.config.model.ResNet_34.filters_imag
            self.kernel_size = self.config.model.ResNet_34.kernel_size
            self.strides = self.config.model.ResNet_34.strides
            self.pool_size = self.config.model.ResNet_34.pool_size
            self.prev_filters_real = self.config.model.ResNet_34.prev_filters_real
            self.prev_filters_imag = self.config.model.ResNet_34.prev_filters_imag
            
        elif(self.network == 'ResNet-50'):
            self.filters_real = self.config.model.ResNet_50.filters_real
            self.filters_imag = self.config.model.ResNet_50.filters_imag
            self.kernel_size = self.config.model.ResNet_50.kernel_size
            self.strides = self.config.model.ResNet_50.strides
            self.pool_size = self.config.model.ResNet_50.pool_size
            self.prev_filters_real = self.config.model.ResNet_50.prev_filters_real
            self.prev_filters_imag = self.config.model.ResNet_50.prev_filters_imag
            
        elif(self.network == 'ResNet-34_2_det'):
            self.filters_real = self.config.model.ResNet_34_2_det.filters_real
            self.filters_imag = self.config.model.ResNet_34_2_det.filters_imag
            self.kernel_size = self.config.model.ResNet_34_2_det.kernel_size
            self.strides = self.config.model.ResNet_34_2_det.strides
            self.pool_size = self.config.model.ResNet_34_2_det.pool_size
            self.prev_filters_real = self.config.model.ResNet_34_2_det.prev_filters_real
            self.prev_filters_imag = self.config.model.ResNet_34_2_det.prev_filters_imag
            
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
        """Loads and Preprocess data """
        
        d_loader = DataLoader(self.n_det, self.dataset, self.num_test, self.n_samples, self.min_snr, self.train_negative_latency, self.train_negative_latency_seconds)
        
        if(self.n_det == 3):
            self.X_train_real, self.X_train_imag = d_loader.load_train_3_det_data(self.config.data, self.snr_range_train, self.train_real, self.psd)
            self.X_test_real, self.X_test_imag = d_loader.load_test_3_det_data(self.config.data, self.test_real, self.snr_range_test, self.psd, self.test_negative_latency, self.test_negative_latency_seconds)
                
            self.y_train, self.intrinsic_train = d_loader.load_train_3_det_parameters(self.config.parameters, self.snr_range_train, self.train_real, self.psd)
            
            self.y_test, self.ra_test, self.dec_test, self.gps_time, self.intrinsic_test = d_loader.load_test_3_det_parameters(self.config.parameters, self.test_real, self.snr_range_test, self.psd, self.test_negative_latency, self.test_negative_latency_seconds)
            
        elif(self.n_det == 2):
            self.X_train_real, self.X_train_imag = d_loader.load_train_2_det_data(self.config.data, self.snr_range_train, self.train_real, self.psd)
            self.X_test_real, self.X_test_imag = d_loader.load_test_2_det_data(self.config.data, self.test_real, self.snr_range_test, self.psd, self.test_negative_latency, self.test_negative_latency_seconds)
                
            self.y_train, self.intrinsic_train = d_loader.load_train_2_det_parameters(self.config.parameters, self.snr_range_train, self.train_real, self.psd)
            
            self.y_test, self.ra_test, self.dec_test, self.gps_time, self.intrinsic_test = d_loader.load_test_2_det_parameters(self.config.parameters, self.test_real, self.snr_range_test, self.psd, self.test_negative_latency, self.test_negative_latency_seconds)
        
        self._preprocess_data(d_loader)
    
    
    def standard_scale_data(self, X_train_real, X_train_imag, X_test_real, X_test_imag):
        
        sc_real = StandardScaler()
        sc_imag = StandardScaler()
        
        X_train_real_h1 = sc_real.fit_transform(self.X_train_real[:,:,0])
        X_train_real_l1 = sc_real.transform(self.X_train_real[:,:,1])
        X_train_real_v1 = sc_real.transform(self.X_train_real[:,:,2])
        
        X_train_imag_h1 = sc_imag.fit_transform(self.X_train_imag[:,:,0])
        X_train_imag_l1 = sc_imag.transform(self.X_train_imag[:,:,1])
        X_train_imag_v1 = sc_imag.transform(self.X_train_imag[:,:,2])
        
        X_test_real_h1 = sc_real.transform(self.X_test_real[:,:,0])
        X_test_real_l1 = sc_real.transform(self.X_test_real[:,:,1])
        X_test_real_v1 = sc_real.transform(self.X_test_real[:,:,2])
        
        X_test_imag_h1 = sc_imag.transform(self.X_test_imag[:,:,0])
        X_test_imag_l1 = sc_imag.transform(self.X_test_imag[:,:,1])
        X_test_imag_v1 = sc_imag.transform(self.X_test_imag[:,:,2])
        
        X_train_real_h1 = X_train_real_h1[:,:,None]
        X_train_real_l1 = X_train_real_l1[:,:,None]
        X_train_real_v1 = X_train_real_v1[:,:,None]
        
        X_train_imag_h1 = X_train_imag_h1[:,:,None]
        X_train_imag_l1 = X_train_imag_l1[:,:,None]
        X_train_imag_v1 = X_train_imag_v1[:,:,None]
        
        X_test_real_h1 = X_test_real_h1[:,:,None]
        X_test_real_l1 = X_test_real_l1[:,:,None]
        X_test_real_v1 = X_test_real_v1[:,:,None]
        
        X_test_imag_h1 = X_test_imag_h1[:,:,None]
        X_test_imag_l1 = X_test_imag_l1[:,:,None]
        X_test_imag_v1 = X_test_imag_v1[:,:,None]       
        
        X_train_real = np.concatenate([X_train_real_h1, X_train_real_l1, X_train_real_v1], axis=-1)
        X_train_imag = np.concatenate([X_train_imag_h1, X_train_imag_l1, X_train_imag_v1], axis=-1)
        
        X_test_real = np.concatenate([X_test_real_h1, X_test_real_l1, X_test_real_v1], axis=-1)
        X_test_imag = np.concatenate([X_test_imag_h1, X_test_imag_l1, X_test_imag_v1], axis=-1)
            
        return X_train_real, X_train_imag, X_test_real, X_test_imag
    
    def standardize_data(self, X_train_real, X_test_real):
        
        X_train_real_mean = np.mean(X_train_real, axis=0)
        X_train_real_std = np.std(X_train_real, axis=0)
        
        X_train_real_standardized = (X_train_real - X_train_real_mean) / X_train_real_std
        X_test_real_standardized = (X_test_real - X_train_real_mean) / X_train_real_std
                
#        X_train_imag_mean = np.mean(X_train_imag, axis=0)
#        X_train_imag_std = np.std(X_train_imag, axis=0)
        
#        X_train_imag_standardized = (X_train_imag - X_train_imag_mean) / X_train_imag_std
#        X_test_imag_standardized = (X_test_imag - X_train_imag_mean) / X_train_imag_std
            
        return X_train_real_standardized, X_test_real_standardized
    
    def scale_labels(self, y_train, y_test):
        
        parameters_mean = np.mean(y_train, axis=0)
        parameters_std = np.std(y_train, axis=0)

        y_train_standardized = (y_train - parameters_mean) / parameters_std
        y_test_standardized = (y_test - parameters_mean) / parameters_std
            
        return y_train_standardized, y_test_standardized, parameters_mean, parameters_std
        
        
    def _preprocess_data(self, d_loader):
        """ Removing < n_det samples and scaling RA and Dec values """
        
#        if((self.n_det == 3) and (self.train_negative_latency == False) and (self.train_real == False) and (self.psd == 'aLIGO')):
#            self.X_train_real, self.X_train_imag, self.y_train, self.ra_x, self.ra_y, self.ra, self.dec, self.h1_snr, self.l1_snr, self.v1_snr = d_loader.load_3_det_samples(self.config.data, self.config.parameters, self.X_train_real, self.X_train_imag, self.y_train, self.num_train, self.snr_range_train, self.snr_range_test, self.psd, data='train')
        
#            if((self.test_real == False) and (self.test_negative_latency == False) and (self.psd == 'aLIGO')):
#                self.X_test_real, self.X_test_imag, self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test, self.h1_snr_test, self.l1_snr_test, self.v1_snr_test = d_loader.load_3_det_samples(self.config.data, self.config.parameters, self.X_test_real, self.X_test_imag, self.y_test, self.num_test, self.snr_range_train, self.snr_range_test, self.psd, data='test')
                
##        elif((self.n_det == 3) and (self.train_negative_latency == False) and (self.train_real == False) and (self.psd == 'design')):
##            self.X_train_real, self.X_train_imag, self.y_train, self.ra_x, self.ra_y, self.ra, self.dec, self.h1_snr, self.l1_snr, self.v1_snr = d_loader.load_3_det_samples(self.config.data, self.config.parameters, self.X_train_real, self.X_train_imag, self.y_train, self.num_train, self.snr_range_train, self.snr_range_test, self.psd, data='train')
        
##            if((self.test_real == False) and (self.test_negative_latency == False) and (self.psd == 'design')):
##                self.X_test_real, self.X_test_imag, self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test, self.h1_snr_test, self.l1_snr_test, self.v1_snr_test = d_loader.load_3_det_samples(self.config.data, self.config.parameters, self.X_test_real, self.X_test_imag, self.y_test, self.num_test, self.snr_range_train, self.snr_range_test, self.psd, data='test')
                
        if((self.psd == 'design') or (self.psd == 'O2') or (self.psd == 'O3')):
            
            self.X_train_real, self.X_train_imag, self.y_train, self.ra, self.ra_x, self.ra_y, self.dec, self.intrinsic_train, self.valid, self.net_snr = d_loader.load_valid_samples(self.X_train_real, self.X_train_imag, self.y_train, self.intrinsic_train, self.train_negative_latency_seconds, self.n_det, data='train')
            
            self.X_test_real, self.X_test_imag, self.y_test, self.ra_test, self.ra_test_x, self.ra_test_y, self.dec_test, self.intrinsic_test, self.valid_test, self.net_snr_test = d_loader.load_valid_samples(self.X_test_real, self.X_test_imag, self.y_test, self.intrinsic_test, self.test_negative_latency_seconds, self.n_det, data='test')
            
        self.gps_time = self.gps_time[self.valid_test]

        self.X_train_real, self.X_test_real = self.standardize_data(self.X_train_real, self.X_test_real)
            
        self.X_train = np.hstack((self.X_train_real, self.X_train_imag))
        shape_train = self.X_train.shape[0]
        self.X_train = self.X_train.reshape(shape_train,2,self.n_samples,self.n_det)
            
        self.X_test = np.hstack((self.X_test_real, self.X_test_imag))
        shape_test = self.X_test.shape[0]
        self.X_test = self.X_test.reshape(shape_test,2,self.n_samples,self.n_det)
            
            
#            noise = np.random.normal(0,1.0,820)
#            noise = noise[None,:,None]
            
#            self.X_train = self.X_train + noise
#            self.X_test = self.X_test + noise
        
        self.y_train, self.y_test, self.mean, self.std = self.scale_labels(self.y_train, self.y_test)
        
            
        
#        elif((self.n_det == 2) and (self.psd == 'design') or (self.psd == 'O2')):
            
#            self.X_train_real, self.X_train_imag, self.y_train, self.ra_x, self.ra_y, self.ra, self.dec, self.h1_snr, self.l1_snr = d_loader.load_2_det_samples(self.config.parameters, self.X_train_real, self.X_train_imag, self.y_train, self.num_train, self.snr_range_train, self.snr_range_test, data='train')
            
#            if(self.test_negative_latency == False):
#                self.X_test_real, self.X_test_imag, self.y_test, self.ra_test_x, self.ra_test_y, self.ra_test, self.dec_test, self.h1_snr_test, self.l1_snr_test = d_loader.load_2_det_samples(self.config.parameters, self.X_test_real, self.X_test_imag, self.y_test, self.num_test, self.snr_range_train, self.snr_range_test, data='test')
           
            
        self.intrinsic_train, self.intrinsic_test, self.intrinsic_mean, self.intrinsic_std = self.scale_labels(self.intrinsic_train, self.intrinsic_test)
            
        
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
            
        self.input1 = tf.keras.layers.Input([2, self.n_samples, self.n_det])
        self.input2 = tf.keras.layers.Input(self.intrinsic_train.shape[-1]) 
        self.x_ = tf.keras.layers.Input(shape=self.y_train.shape[-1], dtype=tf.float32)
            
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
                
#                self.encoded_features = ResNet34(input1, input2, self.filters_real, self.filters_imag, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, self.prev_filters_imag, input_shapes=[self.n_samples, seglf.n_det]).construct_model()
            with strategy.scope():
                self.encoded_features = ResNet34(self.input1, self.input2, self.filters_real, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, input_shapes1=[2, self.n_samples, self.n_det], input_shapes2 = self.intrinsic_train.shape[-1]).construct_model()
        
        elif(self.network == 'ResNet-50'):
                
#                self.encoded_features = ResNet34(input1, input2, self.filters_real, self.filters_imag, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, self.prev_filters_imag, input_shapes=[self.n_samples, self.n_det]).construct_model()
                
            self.encoded_features = ResNet50(input1, input2, self.filters_real, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, input_shapes1=[self.n_samples, self.n_det], input_shapes2 = self.intrinsic_train.shape[-1]).construct_model()
                
        elif(self.network == 'ResNet-34_2_det'):
                
            self.encoded_features = ResNet_34_2_det(input1, input2, self.filters_real, self.filters_imag, self.kernel_size, self.strides, self.pool_size, self.prev_filters_real, self.prev_filters_imag, input_shapes=[self.n_samples, self.n_det]).construct_model()
                
        elif(self.network == 'LSTM'):
                
            self.encoded_features = LSTM_model(input1, input2, self.n_units, self.rate, input_shapes=[None, self.n_det]).construct_model()
                
        elif(self.network == 'CNN'):
                
            self.encoded_features = CNN_model(input1, input2, self.filters, self.kernel_size, self.max_pool_size, self.rate, self.n_units, input_shapes=[self.n_samples, self.n_det]).construct_model()
    
    
    def construct_flow(self, training:bool):
        
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
        
#            self.trainable_distribution = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(2)), bijector = flow_bijector)

#       self.train(log_prob_, checkpoint)

    
    def train(self):
        
        with strategy.scope():
            """Compiles and trains the model"""
            flow = self.construct_flow(training=True)
            self.checkpoint = tf.train.Checkpoint(model=flow)
            
            log_prob_ = -tf.reduce_mean(flow.log_prob(self.x_, bijector_kwargs=
                        self.make_bijector_kwargs(self.trainable_distribution.bijector, 
                                             {'maf.': {'conditional_input':self.encoded_features}})))

            self.model = tf.keras.Model([self.input1, self.input2, self.x_], log_prob_)
    #            self.model = tf.keras.Model([input1, x_], log_prob_)
            self.encoder = tf.keras.Model([self.input1, self.input2], self.encoded_features) 
    #            self.encoder = tf.keras.Model(input1, self.encoded_features)
    
            base_lr = 2e-4
            end_lr = 5e-5
            max_epochs = self.epochs  # maximum number of epochs of the training
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)  # optimizer
    #        self.checkpoint = tf.train.Checkpoint(optimizer=opt, model=self.model)
        
            # load best model with min validation loss
#            self.checkpoint.restore('/fred/oz016/Chayan/GW-SkyNet_pre-merger/checkpoints/BBH_3_det_ResNet-34_adaptive/tmp_0xa9dc39d3/ckpt')
#            self.encoder.load_weights("/fred/oz016/Chayan/GW-SkyNet_pre-merger/model/encoder_models/ResNet-34_BBH_encoder_3_det_adaptive_snr-10to20_test.hdf5")


            custom_checkpoint = CustomCheckpoint(filepath='/fred/oz016/Chayan/GW-SkyLocators/model/encoder_models/'+str(self.network)+'_'+str(self.dataset)+'_encoder_'+str(self.n_det)+'_det_adaptive_snr-10to20_test.hdf5',encoder=self.encoder)
        
            trainingStopCallback = haltCallback(self.model)
        
#        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr, decay_steps=10, decay_rate=0.1)

            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=lambda _, log_prob: log_prob)

#        self.model.compile(optimizer=tf.optimizers.Adam(lr=self.lr), loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)))
    
            self.model.summary()
                                             
        # initialize checkpoints
            dataset_name = "/fred/oz016/Chayan/GW-SkyLocator/checkpoints/"+str(self.dataset)+"_"+str(self.n_det)+"_det_"+str(self.network)+"_adaptive"
            checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
            self.checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=15)
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        
            lr_decayed_fn = SGDRScheduler(min_lr=1e-7,
                                         max_lr=2e-4,
                                         steps_per_epoch=np.ceil(len(self.X_train)/self.batch_size),
                                         lr_decay=0.85,
                                         cycle_length=15,
                                         mult_factor=1.0)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

            callbacks_list=[custom_checkpoint, early_stopping]  

            model_history = self.model.fit([self.X_train, self.intrinsic_train, self.y_train], np.zeros((len(self.X_train_real)), dtype=np.float32),
              batch_size=self.batch_size,
              epochs=self.epochs,
              validation_split=self.val_split,
              callbacks=callbacks_list,
              shuffle=True,
              verbose=True)

#        self.model.fit([self.X_train_real, self.X_train_imag, self.y_train], np.zeros((len(self.X_train_real), 0), dtype=np.float32),
#              batch_size=self.batch_size,
#              epochs=self.epochs,
#              validation_split=self.val_split,
#              callbacks=callbacks_list,
#              shuffle=True,
#              verbose=True)

#            checkpoint.save(file_prefix=checkpoint_prefix)
            self.checkpoint.write(file_prefix=self.checkpoint_prefix)
        
            self.plot_loss_curves(model_history.history['loss'], model_history.history['val_loss'])        
    
    

        # Define the trainable distribution
    def make_masked_autoregressive_flow(self, index, hidden_units, activation, conditional_event_shape):
        
        made = tfp.bijectors.AutoregressiveNetwork(params=2,
                  hidden_units=hidden_units,
                  event_shape=(3,),
                  activation=activation,
                  conditional=True,
    #              kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg', distribution='uniform'),
                  kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1),
    #              kernel_initializer = tf.keras.initializers.RandomNormal(),
                  conditional_event_shape=conditional_event_shape,
                  dtype=np.float32)
    
        return tfp.bijectors.MaskedAutoregressiveFlow(shift_and_log_scale_fn = made, name='maf'+str(index))

    def make_bijector_kwargs(self, bijector, name_to_kwargs):
        
        if hasattr(bijector, 'bijectors'):
            
            return {b.name: self.make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    
        else:
            
            for name_regex, kwargs in name_to_kwargs.items():
                
                if re.match(name_regex, bijector.name):
                    
                    return kwargs
        
        return {}
    
    def scheduler(self, epochs, lr):
        
        if epochs < 50:
            return lr
        elif epochs >= 50:
            return lr * tf.math.exp(-0.1)
    #         return lr * tf.math.exp(-0.1)
           
    #    def train(self, log_prob, checkpoint):
    
    def plot_loss_curves(self, loss, val_loss):
        
        # summarize history for accuracy and loss
        plt.figure(figsize=(6, 4))
        plt.plot(loss, "r--", label="Loss on training data")
        plt.plot(val_loss, "r", label="Loss on validation data")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
            
        plt.savefig("/fred/oz016/Chayan/GW-SkyLocator/evaluation/Loss_curves/Accuracy_curve_GW190814_3_bij_50_epochs.png", dpi=200)
    
        
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
        return z, kde_skl


    #        xy_sample = np.vstack([ra_pix, de_pix])
    #        xy_train  = np.vstack([x, y])
    #        kde = KDE.BoundedKDE(xy_train, bw_method=None)
    #        z = kde.evaluate(xy_sample)
        
    #        return z


    def _bayestar_adaptive_grid(self, preds, kde, top_nside=16, rounds=8):
        
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
            
            print('adaptive refinement round {} of {} ...'.format(
                      iround + 1, rounds - 1))
            cells = sorted(cells, key=lambda p_n_i: p_n_i[0] / p_n_i[1]**2)
            new_nside, new_ipix = np.transpose([
                    (nside * 2, ipix * 4 + i)
                    for _, nside, ipix in cells[-nrefine:] for i in range(4)])
            theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
            ra = phi
            dec = 0.5 * np.pi - theta
            
            xy_sample = np.vstack([dec, ra]).T
            p = np.exp(kde.score_samples(xy_sample))
    #           p = p/(np.sum(p))

            probabilities.append(p)
            
    #            ra = ra[:,None]
    #            dec = dec[:,None]
    #            pixels = np.concatenate([ra, dec], axis=1)

    #            ra = ra - np.pi
    #            ra_x = np.cos(ra)
    #            ra_y = np.sin(ra)
            
    #            pixels = np.column_stack((ra_x, ra_y, dec)).astype('float32')
            
    #            p = self.trainable_distribution.prob((pixels),
#      bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
            
    #            p = self.trainable_distribution.prob(pixels,
#    bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
            
    #            p = p/sum(p)
            
            cells[-nrefine:] = zip(p, new_nside, new_ipix)
        return cells, probabilities
    
    def _bayestar_adaptive_grid_prob_density(self, flow, preds, top_nside=16, rounds=8):
        
        
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
            print('adaptive refinement round {} of {} ...'.format(
                      iround + 1, rounds - 1))
            cells = sorted(cells, key=lambda p_n_i: p_n_i[0]/ p_n_i[1]**2)
            new_nside, new_ipix = np.transpose([
                    (nside * 2, ipix * 4 + i)
                    for _, nside, ipix in cells[-nrefine:] for i in range(4)])
            theta, phi = hp.pix2ang(new_nside, new_ipix, nest=True)
            ra = phi
            dec = 0.5 * np.pi - theta      
                
            ra = ra[:,None]
            dec = dec[:,None]
    #            pixels = np.concatenate([ra, dec], axis=1)

            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            pixels = np.concatenate([ra_x, ra_y, dec], axis=1)
            
    #            pixels = np.dstack((ra_x, ra_y, dec))
            
    #            log_p = self.trainable_distribution.log_prob((pixels),
#      bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
        
    #            min_log_p = tf.math.log(tf.constant(1e-30, dtype=log_p.dtype))
    #            log_p = tf.clip_by_value(log_p, min_log_p, tf.reduce_max(log_p))

    #            p = tf.math.exp(log_p)

            p = flow.prob((pixels),
      bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
        
            p /= tf.reduce_sum(p)
            
            probabilities.append(p)
            
    #            p = self.trainable_distribution.prob(pixels,
#    bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
            
    #            p = p/sum(p)
            
            cells[-nrefine:] = zip(p, new_nside, new_ipix)
        return cells, probabilities
    
    def as_healpix_prob_density(self, flow, preds, top_nside=16):
        """Return a HEALPix multi-order map of the posterior density."""
        prob_MAF = []
        prob_MAF_after_norm = []
        zip_obj, probabilities = self._bayestar_adaptive_grid_prob_density(flow, preds,top_nside=16)
        post, nside, ipix = zip(*zip_obj)
        post = np.asarray(list(post))
        nside = np.asarray(list(nside))
        ipix = np.asarray(list(ipix))

        # Make sure that sky map is normalized (it should be already)
        post /= np.sum(post * ah.nside_to_pixel_area(nside).to_value(u.sr))
    #        post = post/hp.nside2pixarea(nside)
    #        post /= np.sum(post*hp.nside2pixarea(nside))

        # Convert from NESTED to UNIQ pixel indices
        order = np.log2(nside).astype(int)
        uniq = moc.nest2uniq(order.astype(np.int8), ipix)
    #        moc_data = np.rec.fromarrays([uniq, post], names=['UNIQ', 'PROBDENSITY'])

        # Done!
        return Table([uniq, post], names=['UNIQ', 'PROBDENSITY'], copy=False), probabilities
    #        return moc_data

    def as_healpix(self, preds, kde, top_nside=16):
        """Return a HEALPix multi-order map of the posterior density."""
    #        post, nside, ipix = zip(*self._bayestar_adaptive_grid(preds, kde,
    #            top_nside=16))
        zip_obj, probabilities = self._bayestar_adaptive_grid(preds,kde,top_nside=16)
        post, nside, ipix = zip(*zip_obj)
        post = np.asarray(list(post))
        nside = np.asarray(list(nside))
        ipix = np.asarray(list(ipix))
        
        # Make sure that sky map is normalized (it should be already)
        post /= np.sum(post * ah.nside_to_pixel_area(nside).to_value(u.sr))
    #        post = post/hp.nside2pixarea(nside)
    #        post /= np.sum(post*hp.nside2pixarea(nside))

        # Convert from NESTED to UNIQ pixel indices
        order = np.log2(nside).astype(int)
        uniq = moc.nest2uniq(order.astype(np.int8), ipix)
    #        moc_data = np.rec.fromarrays([uniq, post], names=['UNIQ', 'PROBDENSITY'])

        # Done!
        return Table([uniq, post], names=['UNIQ', 'PROBDENSITY'], copy=False), probabilities
    #        return moc_data
    
             
    def obtain_samples(self):
        """Obtain samples from trained distribution"""
        self.encoder.load_weights('/fred/oz016/Chayan/GW-SkyLocator/model/encoder_models/'+str(self.network)+'_'+str(self.dataset)+'_encoder_'+str(self.n_det)+'_det_adaptive_snr-10to20_test.hdf5')
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
    #        de_pix = theta

    #        f1 = h5py.File('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/Adaptive_NSIDE/Negative_latency/Bayestar_comparison_post-merger/New/Injection_run_BNS_3_det_design_test_0_sec_sampling_adaptive_Gaussian_KDE_bandwidth_001_0.hdf', 'r')
    #        p = f1['Probabilities'][()]

        for i in range(self.y_test.shape[0]):
            
            x_test = np.expand_dims(self.X_test[i], axis=0)
            intrinsic_test = np.expand_dims(self.intrinsic_test[i], axis=0)
            
            gps_time_test = self.gps_time[i]
            
    #            x_test = np.expand_dims(self.X_test[i], axis=0)

    #            start = timeit.default_timer()
    
            preds = self.encoder([x_test, intrinsic_test])
            
    #            preds = self.encoder.predict(x_test)
            with strategy.scope():
                
                self.flow = self.construct_flow(training=False)
                checkpoint_test = tf.train.Checkpoint(model=self.flow)
                checkpoint_test.restore(self.checkpoint_prefix)
            
                samples = self.flow.sample((n_samples,),
                  bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
    
    #            samples = self.trainable_distribution.sample((n_samples,),
#              bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
        
    #            print("Time elapsed: {:.5f}".format(timeit.default_timer()-start))
        
                samples = self.std*samples + self.mean
    
                ra_samples_x = samples[:,0]
                ra_samples_y = samples[:,1]
                dec_samples = samples[:,2]

                ra_samples_x = np.where(ra_samples_x > 1, 1, ra_samples_x)
                ra_samples_x = np.where(ra_samples_x < -1, -1, ra_samples_x)
            
                ra_samples_y = np.where(ra_samples_y > 1, 1, ra_samples_y)
                ra_samples_y = np.where(ra_samples_y < -1, -1, ra_samples_y)

                dec_samples = np.where(dec_samples > np.pi/2, np.pi/2, dec_samples)
                dec_samples = np.where(dec_samples < -np.pi/2, -np.pi/2, dec_samples)
            
                ra_samples = np.arctan2(ra_samples_y, ra_samples_x)
                ra_samples = ra_samples + np.pi
            
                detector = Detector('H1')
                delta = detector.gmst_estimate(gps_time_test) - detector.gmst_estimate(gps_time_GW170817)

                ra_samples = np.mod(ra_samples+delta, 2.0*np.pi)

                eps = 1e-5
            
                # A 2D Kernel Density Estimator is used to find the probability density at ra_pix and de_pix
                zz, kde = self.kde2D(ra_samples,dec_samples, 0.02, ra_pix,de_pix) # for 0 sec: 0.01, for -15 sec, -30 sec: 0.04
                zz = zz/(np.sum(zz))
            
                # Uncomment for everything except 0 secs        
    #            hpmap, probability = self.as_healpix(zz,kde)
            
                # Post-merger injection run 
                io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/BBH/skymaps/Gaussian_KDE/Test_new_BN_test_GPU_'+str(i)+'.fits', hpmap, nest=True)

                # Pre-merger injection run
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/45_secs/Test_3_bij_50_epochs_lr_schedule_'+str(i)+'.fits', hpmap, nest=True)
        
                # Real events
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Real_events/Test_3_bij_50_epochs_NSBH_3_det_new_BN_'+str(i)+'.fits', hpmap, nest=True)

                # Bayestar example
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_test/evaluation/skymaps/CPU/Bayestar_test/Test_Bayestar_example_6_'+str(i)+'.fits', hpmap, nest=True)

                probs.append(zz)
            
                ra_preds.append(ra_samples)
                dec_preds.append(dec_samples)
            
    #            probabilities.append(probability)
        
            self.ra_test = self.ra_test + np.pi
            
        
        if(self.n_det == 3):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
#            f1.create_dataset('Probabilities_adaptive', data = probabilities)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            f1.create_dataset('Valid', data = self.valid_test)
            f1.create_dataset('Network_SNR', data = self.net_snr_test)
#            f1.create_dataset('H1_SNR', data = self.h1_snr_test)
#            f1.create_dataset('L1_SNR', data = self.l1_snr_test)
#            f1.create_dataset('V1_SNR', data = self.v1_snr_test)

            f1.close()    
        
        elif(self.n_det == 2):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            
            f1.close()
                
            
    def obtain_probability_density(self):
        """Obtain probability density from trained distribution"""
        self.encoder.load_weights('/fred/oz016/Chayan/GW-SkyLocator/model/encoder_models/'+str(self.network)+'_'+str(self.dataset)+'_encoder_'+str(self.n_det)+'_det_adaptive_snr-10to20_test.hdf5')

        probs = []
        preds_array = []
        probabilities = []
        ra_preds = []
        dec_preds = []
        
        nside=16
        deg2perpix = hp.nside2pixarea(nside)
        npix=hp.nside2npix(nside)
        theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))

        # ra_pix and de_pix are co-ordinates in the skuy where I want to find the probabilities
        ra_pix = phi
        de_pix = -theta + np.pi/2.0 
    #        de_pix = theta
        
        ra_pix = ra_pix - np.pi
        ra_pix_x = np.cos(ra_pix)
        ra_pix_y = np.sin(ra_pix)
         
        ra_pix = ra_pix[:,None]
        ra_pix_x = ra_pix_x[:,None]
        ra_pix_y = ra_pix_y[:,None]
    ##        ra_pix = ra_pix[:,None]
        de_pix = de_pix[:,None]

        pixels = np.concatenate([ra_pix_x, ra_pix_y, de_pix], axis=1)
    #        pixels = np.dstack((ra_pix_x, ra_pix_y, de_pix))

    #        mean_pixels = np.mean(pixels, axis=0)
    #        std_pixels = np.std(pixels, axis=0)
    #        pixels = (pixels - mean_pixels)/std_pixels
    #        pixels = tf.cast(pixels, tf.float32)
        
        eps = 1e-5
       

        for i in range(self.y_test.shape[0]):
            
            x_test = np.expand_dims(self.X_test[i], axis=0)
            intrinsic_test = np.expand_dims(self.intrinsic_test[i], axis=0)
            
    #            start = timeit.default_timer()
    
            preds = self.encoder([x_test, intrinsic_test])
            preds_array.append(preds)


            self.flow = self.construct_flow(training=False)
            checkpoint_test = tf.train.Checkpoint(model=self.flow)
            checkpoint_test.restore(self.checkpoint_prefix)
            
    #            prob_density = tf.math.exp(self.trainable_distribution.log_prob(pixels, bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}})))
            
            prob_density = tf.math.exp(self.flow.log_prob(pixels, bijector_kwargs=self.make_bijector_kwargs(self.trainable_distribution.bijector, {'maf.': {'conditional_input':preds}})))
            
            

            # Uncomment for everything except 0 secs        
            hpmap, probability = self.as_healpix_prob_density(self.flow,preds)
    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/NSBH/skymaps/Gaussian_KDE/Test_3_bij_lr_schedule_'+str(i)+'.fits', hpmap, nest=True)

    #            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/15_secs/Test_3_bij_50_epochs_prob_density_'+str(i)+'.fits', hpmap, nest=True)
            io.fits.write_sky_map('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/Real_events/Test_3_bij_50_epochs_BNS_3_det_prob_density_'+str(i)+'.fits', hpmap, nest=True)
    
            probabilities.append(probability)
                 
                        
        self.ra_test = self.ra_test + np.pi
        
        if(self.n_det == 3):
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
            f1.create_dataset('Probabilities_adaptive', data = probabilities)
            f1.create_dataset('Preds', data = preds_array)
            f1.create_dataset('RA_samples', data = ra_preds)
            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            f1.create_dataset('Valid', data = self.valid_test)
    #            f1.create_dataset('H1_SNR', data = self.h1_snr_test)
    #            f1.create_dataset('L1_SNR', data = self.l1_snr_test)
    #            f1.create_dataset('V1_SNR', data = self.v1_snr_test)

            f1.close()    
        
        elif(self.n_det == 2):
            
            f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/'+self.output_filename, 'w')
            f1.create_dataset('Probabilities', data = probs)
    #            f1.create_dataset('RA_samples', data = ra_preds)
    #            f1.create_dataset('Dec_samples', data = dec_preds)
            f1.create_dataset('RA_test', data = self.ra_test)
            f1.create_dataset('Dec_test', data = self.dec_test)
            
            f1.close()
                    
