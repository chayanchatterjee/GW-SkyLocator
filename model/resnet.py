# Imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

class ResNet:
    
    def __init__(self, input1, input2, kernels_res, 
                 kernel_size_res, stride_res, kernels, kernel_size, strides):
        
        self.input1 = input1
        self.input2 = input2
        self.kernels_res = kernels_res
        self.stride_res = stride_res
        self.kernel_size_res = kernel_size_res
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.strides = strides
        
    def residual_block(self, X, kernels_res, stride_res, kernel_size_res):
        
        out = tf.keras.layers.BatchNormalization()(X)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv1D(kernels_res, kernel_size_res, stride_res, padding='same')(X)
    
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv1D(kernels_res, kernel_size_res, stride_res, padding='same')(out)
        out = tf.keras.layers.add([X, out])
        
        return out
        
    def construct_model(self):
        
        # Real part
        
        X = tf.keras.layers.Conv1D(self.kernels, self.strides)(self.input1)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)

        flat1 = tf.keras.layers.Flatten()(X)
        

        # Imaginary part
        
        X = tf.keras.layers.Conv1D(self.kernels, self.strides)(self.input2)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)
        X = residual_block(X, self.kernels_res, self.strides_res)
        
        X = tf.keras.layers.Conv1D(self.kernels, kernel_size=self.kernel_size, strides=self.strides, padding='same')(X)

        flat2 = tf.keras.layers.Flatten()(X)

   
        # merge input models
    
        merge = tf.keras.layers.concatenate([flat1, flat2])
        
        return merge
        
        
        
        
        