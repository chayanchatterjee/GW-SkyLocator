# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

class CNN_model:
    def __init__(self, input1, input2, filters, kernel_size, max_pool_size, dropout_rate, n_units, input_shapes):
        
        self.input1 = input1
        self.input2 = input2
        self.filters = filters
        self.kernel_size = kernel_size
        self.max_pool_size = max_pool_size
        self.rate = dropout_rate
        self.input_shapes = input_shapes
        self.n_units = n_units
        
    def construct_model(self):
        
        # Real part
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shapes)(self.input1)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.MaxPooling1D(self.max_pool_size)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.MaxPooling1D(self.max_pool_size)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.MaxPooling1D(self.max_pool_size)(X)
                
        X = tf.keras.layers.Flatten()(X)
        
        dense1_real = tf.keras.layers.Dense(self.n_units, activation='relu')(X)
        dense2_real = tf.keras.layers.Dense(self.n_units/2, activation='relu')(dense1_real)
        
        
        # Imaginary part
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shapes)(self.input2)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.MaxPooling1D(self.max_pool_size)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.MaxPooling1D(self.max_pool_size)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.MaxPooling1D(self.max_pool_size)(X)
                
        X = tf.keras.layers.Flatten()(X)
        
        dense1_imag = tf.keras.layers.Dense(self.n_units, activation='relu')(X)
        dense2_imag = tf.keras.layers.Dense(self.n_units/2, activation='relu')(dense1_imag)
        
        
                
        # Merge features
        
        merge = tf.keras.layers.concatenate([dense2_real, dense2_imag])
        
        return merge
        
