# Imports
import numpy as np
import tensorflow as tf

class SimulatedAnnealingCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, loss, initial_temperature=1.0, schedule=None):
        super().__init__()
        self.loss = loss
        self.temperature = initial_temperature
        self.schedule = schedule or np.logspace(-2, 0, num=100, endpoint=True)[::-1]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < len(self.schedule):
            self.epoch = epoch
            self.temperature = self.schedule[epoch]

    def on_train_batch_begin(self, batch, logs=None):
        annealing_factor = self.schedule[min(self.epoch, len(self.schedule)-1)]
        annealed_loss = annealing_factor * self.loss / self.temperature
        logs['loss'] = -tf.reduce_mean(tf.exp(-annealed_loss))
        
        return