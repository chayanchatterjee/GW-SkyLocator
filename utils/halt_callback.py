# Imports
import numpy as np
import tensorflow as tf

class haltCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, model):
        
        self.monitor = 'val_loss'
        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if(current <= -35.0):
            print("\n\n\nReached -35.0 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True