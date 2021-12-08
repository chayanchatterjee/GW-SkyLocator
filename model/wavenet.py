# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

class WaveNet:
    
    def __init__(self, input1, input2, filters, kernel_size, activation, dilation_rate):
        
        self.input1 = input1
        self.input2 = input2
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.dilation_rate = dilation_rate
        
    def construct_model(self):
        
        # Real part
        
        X = tf.keras.layers.Conv1D(filters = self.filters, kernel_size = self.kernel_size, padding='causal', 
                                       activation=self.activation, dilation_rate=self.dilation_rate)(self.input1)
        
        X = tf.keras.layers.Conv1D(filters = self.filters, kernel_size = self.kernel_size, padding='causal', 
                                       activation=self.activation, dilation_rate=2*self.dilation_rate )(X)
            
        X = tf.keras.layers.Conv1D(filters = self.filters, kernel_size = self.kernel_size, padding='causal', 
                                       activation=self.activation, dilation_rate=4*self.dilation_rate )(X)
        
        X = tf.keras.layers.Conv1D(filters = self.filters, kernel_size = self.kernel_size, padding='causal', 
                                       activation=self.activation, dilation_rate=8*self.dilation_rate )(X)
        
        X = tf.keras.layers.Conv1D(filters = self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate= self.dilation_rate)(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=2*self.dilation_rate)(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=4*self.dilation_rate)(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=8*self.dilation_rate)(X)
        
        flat1 = tf.keras.layers.Flatten()(X)
        
        # Imaginary part
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=self.dilation_rate)( self.input2)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=2*self.dilation_rate)(X)
            
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=4*self.dilation_rate)(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=8*self.dilation_rate)(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate= self.dilation_rate)(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=2*self.dilation_rate )(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=4*self.dilation_rate)(X)
        
        X = tf.keras.layers.Conv1D(filters =  self.filters, kernel_size =  self.kernel_size, padding='causal', 
                                       activation= self.activation, dilation_rate=8*self.dilation_rate)(X)
        
        flat2 = tf.keras.layers.Flatten()(X)
        
        # merge input models
        
        merge = tf.keras.layers.concatenate([flat1, flat2])    
    
        return merge
    
    
    
    
    
