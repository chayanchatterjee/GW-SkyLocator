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

# external

from __future__ import print_function
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
import tensorflow as tf
import healpy as hp

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
        
        self.num_train = self.config.train.num_train
        self.num_test = self.config.train.num_test
        self.n_samples = self.config.train.n_samples
        self.n_det = self.config.train.num_detectors
        self.epochs = self.config.train.epochs
        
        if(self.network == 'WaveNet'):
            self.filters = self.config.model.WaveNet.filters
            self.kernel_size = self.config.model.WaveNet.kernel_size
            self.activation = self.config.model.WaveNet.activation
            
        elif(self.network == 'ResNet'):
            self.kernels_res = self.config.model.ResNet.kernels_resnet_block
            self.stride_res = self.config.model.ResNet.stride_resent_block
            self.kernel_size_res = self.config.model.ResNet.kernel_size_resnet_block
            self.kernel_size = self.config.model.ResNet.kernel_size
            self.strides = self.config.model.ResNet.strides
            
        self.num_bijectors = self.config.model.num_bijectors
        self.MAF_hidden_units = self.config.model.MAF_hidden_units
        
        self.lr = self.config.model.learning_rate
        
    def load_data(self):
         """Loads and Preprocess data """
          
            
            
            
        
        
            
    




