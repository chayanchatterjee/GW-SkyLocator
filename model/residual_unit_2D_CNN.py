# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu',**kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(self.filters, 3, strides=strides, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(self.filters, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()]
        
        self.skip_layers = []
        if (strides > 1):
            self.skip_layers = [
                tf.keras.layers.Conv2D(self.filters, 1, strides=strides, padding='same', use_bias=False),
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
        
        
        
        
        
