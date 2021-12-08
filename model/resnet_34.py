# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

from .residual_unit import ResidualUnit

class ResNet34(ResidualUnit):
    def __init__(self, input1, input2, filters, kernel_size, strides, pool_size, prev_filters, input_shapes):
        
        self.input1 = input1
        self.input2 = input2
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.prev_filters = prev_filters
        self.input_shapes = input_shapes
        
    def construct_model(self):
        
        # Real part
        
        X = tf.keras.layers.Conv1D(self.filters, self.kernel_size, strides=self.strides, input_shape=self.input_shapes,
                                   padding='same', use_bias=False)(self.input1)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, strides=self.strides, padding='same')(X)
        
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == self.prev_filters else 2
            X = ResidualUnit(filters, strides=strides)(X)
            self.prev_filters = filters
        X = tf.keras.layers.GlobalAvgPool1D()(X)
        flat1 = tf.keras.layers.Flatten()(X)
        
        
        # Imaginary part
        
        X = tf.keras.layers.Conv1D(self.filters, self.kernel_size, strides=self.strides, input_shape=self.input_shapes,
                                   padding='same', use_bias=False)(self.input2)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPool1D(pool_size=self.pool_size, strides=self.strides, padding='same')(X)
        
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == self.prev_filters else 2
            X = ResidualUnit(filters, strides=strides)(X)
            self.prev_filters = filters
        X = tf.keras.layers.GlobalAvgPool1D()(X)
        flat2 = tf.keras.layers.Flatten()(X)
        
        
        # Merge features
        
        merge = tf.keras.layers.concatenate([flat1, flat2])
        
        return merge
        
    