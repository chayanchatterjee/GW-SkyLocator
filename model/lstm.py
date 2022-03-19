# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

class LSTM_model:
    def __init__(self, input1, input2, n_units, rate, input_shapes):
        
        self.input1 = input1
        self.input2 = input2
        self.n_units = n_units
        self.rate = rate
        self.input_shapes = input_shapes
        
    def construct_model(self):
        
        # Real part
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True, return_sequences=True, input_shape=self.input_shapes)(self.input1)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        flat1 = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=False)(X)
#        X = tf.keras.layers.Dropout(self.rate)(X)
        
#        flat1 = tf.keras.layers.Flatten()(X)
        
        
        # Imaginary part
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True, return_sequences=True, input_shape=self.input_shapes)(self.input2)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        X = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dropout(self.rate)(X)
        
        flat2 = tf.keras.layers.LSTM(self.n_units, use_bias=True,  return_sequences=False)(X)
#        X = tf.keras.layers.Dropout(self.rate)(X)
        
#        flat2 = tf.keras.layers.Flatten()(X)
        
                
        # Merge features
        
        merge = tf.keras.layers.concatenate([flat1, flat2])
        
        return merge
        
