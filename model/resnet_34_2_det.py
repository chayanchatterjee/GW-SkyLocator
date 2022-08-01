# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

from .residual_unit import ResidualUnit

class ResNet_34_2_det(ResidualUnit):
    def __init__(self, input1, input2, filters_real, filters_imag, kernel_size, strides, pool_size, prev_filters_real, prev_filters_imag, input_shapes):
        
        self.input1 = input1
        self.input2 = input2
        self.filters_real = filters_real
        self.filters_imag = filters_imag
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.prev_filters_real = prev_filters_real
        self.prev_filters_imag = prev_filters_imag
        self.input_shapes = input_shapes
        
    def construct_model(self):
        
        # Real part
        
        X_real = tf.keras.layers.Conv1D(self.filters_real, self.kernel_size, strides=self.strides, input_shape=self.input_shapes,
                                   padding='same', use_bias=False)(self.input1)
        X_real = tf.keras.layers.BatchNormalization()(X_real)
        X_real = tf.keras.layers.Activation('relu')(X_real)
#        X_real = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, strides=self.strides, padding='same')(X_real)
        X_real = tf.keras.layers.Conv1D(self.filters_real, self.kernel_size, strides=self.strides,
                                   padding='same', use_bias=False)(X_real)
        
        for filters in [self.filters_real] * 3 + [self.filters_real*2] * 4 + [self.filters_real*4] * 4 + [self.filters_real*8] * 3:
#        for filters in [self.filters_real] * 2 + [self.filters_real*2] * 2 + [self.filters_real*4] * 2 + [self.filters_real*8] * 2:
            strides = 1 if filters == self.prev_filters_real else 2
            X_real = ResidualUnit(filters, strides=strides)(X_real)
            self.prev_filters_real = filters
            
        X_real = tf.keras.layers.GlobalAvgPool1D()(X_real)
        flat1 = tf.keras.layers.Flatten()(X_real)
#        dense1_real = tf.keras.layers.Dense(units=128, activation='relu')(flat1)
#        dense2_real = tf.keras.layers.Dense(units=64, activation='relu')(flat1)

#        drop1 = tf.keras.layers.Dropout(rate=0.15)(flat1)
        
        # Imaginary part
        
        X_imag = tf.keras.layers.Conv1D(self.filters_imag, self.kernel_size, strides=self.strides, input_shape=self.input_shapes,
                                   padding='same', use_bias=False)(self.input2)
        X_imag = tf.keras.layers.BatchNormalization()(X_imag)
        X_imag = tf.keras.layers.Activation('relu')(X_imag)
#        X_imag = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, strides=self.strides, padding='same')(X_imag)
        X_imag = tf.keras.layers.Conv1D(self.filters_imag, self.kernel_size, strides=self.strides,
                                   padding='same', use_bias=False)(X_imag)
        
        for filters in [self.filters_imag] * 3 + [self.filters_imag*2] * 4 + [self.filters_imag*4] * 4 + [self.filters_imag*8] * 3:
#        for filters in [self.filters_imag] * 2 + [self.filters_imag*2] * 2 + [self.filters_imag*4] * 2 + [self.filters_imag*8] * #2:
            strides = 1 if filters == self.prev_filters_imag else 2
            X_imag = ResidualUnit(filters, strides=strides)(X_imag)
            self.prev_filters_imag = filters
            
        X_imag = tf.keras.layers.GlobalAvgPool1D()(X_imag)
        flat2 = tf.keras.layers.Flatten()(X_imag)
#        dense1_imag = tf.keras.layers.Dense(units=128, activation='relu')(flat2)
#        dense2_imag = tf.keras.layers.Dense(units=64, activation='relu')(flat2)

#        drop2 = tf.keras.layers.Dropout(rate=0.15)(flat2)
    
        # Merge features
        
#        merge = tf.keras.layers.concatenate([dense1_real, dense1_imag], axis=-1)

        merge = tf.keras.layers.concatenate([flat1, flat2])
        
        # Flatten features
        
#        flat = tf.keras.layers.Flatten()(merge)
        
        dense1 = tf.keras.layers.Dense(units=256, activation='relu')(merge)
#        dense2 = tf.keras.layers.Dense(units=128, activation='relu')(dense1)
        
#        return flat
        
        return dense1
