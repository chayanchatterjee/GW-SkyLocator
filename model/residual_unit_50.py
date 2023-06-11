# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

class ResidualUnit50(tf.keras.layers.Layer):
    def __init__(self, filters, stage, strides, activation='relu',**kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stage = stage
        self.strides = strides
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv1D(self.filters, 1, strides=self.strides, padding='valid', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv1D(self.filters, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv1D(4*self.filters, 1, strides=1, padding='valid', use_bias=False),
            tf.keras.layers.BatchNormalization()]
        
        self.skip_layers = []
        if ((self.strides > 1) or (self.stage == 2)):
            self.skip_layers = [
                tf.keras.layers.Conv1D(4*self.filters, 1, strides=self.strides, padding='valid', use_bias=False),
                tf.keras.layers.BatchNormalization()]
            
            
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'filters': self.filters
        })
        return config
        
        
        
        
        