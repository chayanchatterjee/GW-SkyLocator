# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

from .residual_unit_50 import ResidualUnit50

class ResNet50(ResidualUnit50):
    def __init__(self, input1, input2, filters_real, kernel_size, strides, pool_size, prev_filters_real, input_shapes1, input_shapes2):
        
        self.input1 = input1
        self.input2 = input2
        self.filters_real = filters_real
#        self.filters_imag = filters_imag
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.prev_filters_real = prev_filters_real
#        self.prev_filters_imag = prev_filters_imag
        self.input_shapes1 = input_shapes1
        self.input_shapes2 = input_shapes2
        
    def construct_model(self):
        
        # Real part
        
        X = tf.keras.layers.Conv1D(self.filters_real, self.kernel_size, strides=self.strides, input_shape=self.input_shapes1,
                                   padding='same', use_bias=False)(self.input1)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, strides=self.strides, padding='same')(X)
        
        stage = 2
        for filters in [self.filters_real] * 3 + [2*self.filters_real] * 4 + [4*self.filters_real] * 6 + [8*self.filters_real] * 3:
            
            strides = 1 if filters == self.prev_filters_real else 2
            X = ResidualUnit50(filters, stage, strides=strides)(X)
            self.prev_filters_real = filters
            stage = stage+1

#        flat1 = tf.keras.layers.Flatten()(X) 
        
        resnet1 = tf.keras.layers.GlobalAvgPool1D()(X)
 #       flat1 = tf.keras.layers.Flatten()(resnet1) 
        
        dense_1 = tf.keras.layers.Dense(units=128, activation='relu')(resnet1)
        dense_1 = tf.keras.layers.BatchNormalization()(dense_1)
        dense_2 = tf.keras.layers.Dense(units=64, activation='relu')(dense_1)
#        dense_real_3 = tf.keras.layers.Dense(units=128, activation='relu')(dense_real_2)
#        dense_real_4 = tf.keras.layers.Dense(units=32, activation='relu')(dense_real_3)
#        dense_real_5 = tf.keras.layers.Dense(units=16, activation='relu')(dense_real_4)
#        flat1 = tf.keras.layers.Flatten()(X)
        
        
        # Imaginary part
        
#        X = tf.keras.layers.Conv1D(self.filters_imag, self.kernel_size, strides=self.strides, input_shape=self.input_shapes,
#                                   padding='valid', use_bias=False)(self.input2)
#        X = tf.keras.layers.BatchNormalization()(X)
#        X = tf.keras.layers.Activation('relu')(X)
#        X = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, padding='valid')(X)
        
#        for filters in [self.filters_imag] * 3 + [2*self.filters_imag] * 4 + [4*self.filters_imag] * 6 + [8*self.filters_imag] * 3:
#            strides = 1 if filters == self.prev_filters_imag else 2
#            X = ResidualUnit(filters, strides=strides)(X)
#            self.prev_filters_imag = filters
            
#        flat2 = tf.keras.layers.Flatten()(X) 
            
##        X = tf.keras.layers.GlobalAvgPool1D()(X)
#        dense_imag_1 = tf.keras.layers.Dense(units=256, activation='relu')(flat2)
#        dense_imag_2 = tf.keras.layers.Dense(units=128, activation='relu')(dense_imag_1)
#        dense_imag_3 = tf.keras.layers.Dense(units=64, activation='relu')(dense_imag_2)
##        dense_imag_4 = tf.keras.layers.Dense(units=32, activation='relu')(dense_imag_3)
##        dense_imag_5 = tf.keras.layers.Dense(units=16, activation='relu')(dense_imag_4)
##        flat2 = tf.keras.layers.Flatten()(X)
                
        # Merge features
        
#        merge = tf.keras.layers.concatenate([dense_real_3, dense_imag_3], axis=-1)


        # Intrinsic parameters
    
        dense_real_1 = tf.keras.layers.Dense(units=64, activation='relu')(self.input2)
        dense_real_1 = tf.keras.layers.BatchNormalization()(dense_real_1)
        dense_real_2 = tf.keras.layers.Dense(units=64, activation='relu')(dense_real_1)
        dense_real_2 = tf.keras.layers.BatchNormalization()(dense_real_2)
        dense_real_3 = tf.keras.layers.Dense(units=64, activation='relu')(dense_real_2)
        dense_real_3 = tf.keras.layers.BatchNormalization()(dense_real_3)
#        dropout1 = tf.keras.layers.Dropout(rate=0.1)(dense_real_3)
        dense_real_4 = tf.keras.layers.Dense(units=64, activation='relu')(dense_real_3)
        dense_real_4 = tf.keras.layers.BatchNormalization()(dense_real_4)
        dense_real_5 = tf.keras.layers.Dense(units=64, activation='relu')(dense_real_4)
        
        # Merge features
        
        merge = tf.keras.layers.concatenate([dense_2, dense_real_5], axis=-1)
        
        
        return merge
        
    
