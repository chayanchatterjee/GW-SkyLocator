from __future__ import print_function
from matplotlib import pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools

from matplotlib import pyplot as plt

#%matplotlib inline
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
#tf.enable_eager_execution()
import healpy as hp

device_type = 'GPU'
n_gpus = 2
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])

import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile

import numpy as np
import pandas as pd
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn
import itertools
#from tensorflow_addons.optimizers import CyclicalLearningRate
#import matplotlib as mpl
#mpl.style.use('seaborn')

# Reading the SNR time series hdf files for training

import random
import os

import h5py

f1 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_52k.hdf', 'r')
f2 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_30k.hdf', 'r')
f3 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_42-47.hdf', 'r')
f4 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_48-50.hdf', 'r')

h1_real_52k = abs(f1['h1_snr_series'][()])
l1_real_52k = abs(f1['l1_snr_series'][()])
v1_real_52k = abs(f1['v1_snr_series'][()])
h1_real_30k = abs(f2['h1_snr_series'][()])
l1_real_30k = abs(f2['l1_snr_series'][()])
v1_real_30k = abs(f2['v1_snr_series'][()])
h1_real_12k = abs(f3['h1_snr_series'][()])
l1_real_12k = abs(f3['l1_snr_series'][()])
v1_real_12k = abs(f3['v1_snr_series'][()])
h1_real_6k = abs(f4['h1_snr_series'][()])
l1_real_6k = abs(f4['l1_snr_series'][()])
v1_real_6k = abs(f4['v1_snr_series'][()])


h1_real = np.concatenate([h1_real_52k, h1_real_30k, h1_real_12k, h1_real_6k], axis=0)
l1_real = np.concatenate([l1_real_52k, l1_real_30k, l1_real_12k, l1_real_6k], axis=0)
v1_real = np.concatenate([v1_real_52k, v1_real_30k, v1_real_12k, v1_real_6k], axis=0)

h1_imag_52k = np.imag(f1['h1_snr_series'][()])
l1_imag_52k = np.imag(f1['l1_snr_series'][()])
v1_imag_52k = np.imag(f1['v1_snr_series'][()])
h1_imag_30k = np.imag(f2['h1_snr_series'][()])
l1_imag_30k = np.imag(f2['l1_snr_series'][()])
v1_imag_30k = np.imag(f2['v1_snr_series'][()])
h1_imag_12k = np.imag(f3['h1_snr_series'][()])
l1_imag_12k = np.imag(f3['l1_snr_series'][()])
v1_imag_12k = np.imag(f3['v1_snr_series'][()])
h1_imag_6k = np.imag(f4['h1_snr_series'][()])
l1_imag_6k = np.imag(f4['l1_snr_series'][()])
v1_imag_6k = np.imag(f4['v1_snr_series'][()])

h1_imag = np.concatenate([h1_imag_52k, h1_imag_30k, h1_imag_12k, h1_imag_6k], axis=0)
l1_imag = np.concatenate([l1_imag_52k, l1_imag_30k, l1_imag_12k, l1_imag_6k], axis=0)
v1_imag = np.concatenate([v1_imag_52k, v1_imag_30k, v1_imag_12k, v1_imag_6k], axis=0)

f1.close()
f2.close()
f3.close()
f4.close()

f1 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_52k.hdf', 'r')
f2 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_30k.hdf', 'r')
f3 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_42-47.hdf', 'r')
f4 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_48-50.hdf', 'r')

ra_52k = 2.0*np.pi*f1['ra'][()]
dec_52k = np.arcsin(1.0-2.0*f1['dec'][()])
ra_30k = 2.0*np.pi*(f2['ra'][0:30000][()])
dec_30k = np.arcsin(1.0-2.0*f2['dec'][0:30000][()])
ra_12k = 2.0*np.pi*(f3['ra'][()])
dec_12k = np.arcsin(1.0-2.0*f3['dec'][()])
ra_6k = 2.0*np.pi*(f4['ra'][()])
dec_6k = np.arcsin(1.0-2.0*f4['dec'][()])

ra = np.concatenate([ra_52k, ra_30k, ra_12k, ra_6k])
dec = np.concatenate([dec_52k, dec_30k, dec_12k, dec_6k])

f1.close()
f2.close()
f3.close()
f4.close()


from SampleFileTools1 import SampleFile

#dec = -dec + np.pi/2.0

# Reading test set data

#f_test = h5py.File('SNR_time_series_sample_files/default_GW170817_injection_run_parameters.hdf', 'r')



f_test = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_NSBH_test.hdf', 'r')

data_ra = f_test['ra'][()]
data_dec = f_test['dec'][()]
#data_v1_test = group_test['v1_snr']
    
f_test.close()

ra_test = 2.0*np.pi*data_ra
dec_test = np.arcsin(1.0 - 2.0*data_dec)

#dec_test = -dec_test + np.pi/2.0

#f3 = h5py.File('samplegen/output/default_snr_series_GW170817_snr_test_injection_run.hdf', 'r')

f3 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_test.hdf', 'r')

#Get the HDF5 group
group3 = f3['omf_injection_snr_samples']

data_h1_3 = group3['h1_snr']
data_l1_3 = group3['l1_snr']
data_v1_3 = group3['v1_snr']

num_test_samples = 2000

index = np.arange(num_test_samples) 

existing_index = [i for i in index if str(i) in data_h1_3.keys()]   

h1_test_real = np.zeros((len(existing_index),410))
l1_test_real = np.zeros((len(existing_index),410))
v1_test_real = np.zeros((len(existing_index),410))

h1_test_imag = np.zeros((len(existing_index),410))
l1_test_imag = np.zeros((len(existing_index),410))
v1_test_imag = np.zeros((len(existing_index),410))

ra_test_new = np.zeros(len(existing_index))
dec_test_new = np.zeros(len(existing_index))

for i,j in zip(range(len(existing_index)),existing_index):
    h1_test_real[i] = abs(data_h1_3[str(j)][()][1840:2250])
    l1_test_real[i] = abs(data_l1_3[str(j)][()][1840:2250])
    v1_test_real[i] = abs(data_v1_3[str(j)][()][1840:2250])
    
    h1_test_imag[i] = np.imag(data_h1_3[str(j)][()][1840:2250])
    l1_test_imag[i] = np.imag(data_l1_3[str(j)][()][1840:2250])
    v1_test_imag[i] = np.imag(data_v1_3[str(j)][()][1840:2250])
    
    ra_test_new[i] = ra_test[j]
    dec_test_new[i] = dec_test[j]
    

from sklearn.preprocessing import StandardScaler
#sc_h1_real = StandardScaler()
#sc_l1_real = StandardScaler()
#sc_v1_real = StandardScaler()

#sc_h1_imag = StandardScaler()
#sc_l1_imag = StandardScaler()
#sc_v1_imag = StandardScaler()

#h1_real = sc_h1_real.fit_transform(h1_real)
#l1_real = sc_l1_real.fit_transform(l1_real)
#v1_real = sc_v1_real.fit_transform(v1_real)

#h1_imag = sc_h1_imag.fit_transform(h1_imag)
#l1_imag = sc_l1_imag.fit_transform(l1_imag)
#v1_imag = sc_v1_imag.fit_transform(v1_imag)

h1_real = h1_real[:,:,None]
l1_real = l1_real[:,:,None]
v1_real = v1_real[:,:,None]

h1_imag = h1_imag[:,:,None]
l1_imag = l1_imag[:,:,None]
v1_imag = v1_imag[:,:,None]


X_train_real = np.concatenate((h1_real, l1_real, v1_real), axis=2)
X_train_imag = np.concatenate((h1_imag, l1_imag, v1_imag), axis=2)

#h1_test_real = sc_h1_real.transform(h1_test_real)
#l1_test_real = sc_l1_real.transform(l1_test_real)
#v1_test_real = sc_v1_real.transform(v1_test_real)

#h1_test_imag = sc_h1_imag.transform(h1_test_imag)
#l1_test_imag = sc_l1_imag.transform(l1_test_imag)
#v1_test_imag = sc_v1_imag.transform(v1_test_imag)

h1_test_real = h1_test_real[:,:,None]
l1_test_real = l1_test_real[:,:,None]
v1_test_real = v1_test_real[:,:,None]

h1_test_imag = h1_test_imag[:,:,None]
l1_test_imag = l1_test_imag[:,:,None]
v1_test_imag = v1_test_imag[:,:,None]


X_test_real = np.concatenate((h1_test_real, l1_test_real, v1_test_real), axis=2)
X_test_imag = np.concatenate((h1_test_imag, l1_test_imag, v1_test_imag), axis=2)

ra = ra[:,None]
dec = dec[:,None]

y_train = np.concatenate((ra, dec), axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#sc_train_real = StandardScaler()
#sc_train_imag = StandardScaler()

y_train = sc.fit_transform(y_train)
#X_train_real = sc_train_real.fit_transform(X_train_real)
#X_train_imag = sc_train_imag.fit_transform(X_train_imag)

ra_test_new = ra_test_new[:,None]
dec_test_new = dec_test_new[:,None]

y_test = np.concatenate((ra_test_new, dec_test_new), axis=1)

y_test = sc.transform(y_test)
#X_test_real = sc_train_real.transform(X_test_real)
#X_test_imag = sc_train_imag.transform(X_test_imag)

f3.close()

# Convert type for Keras otherwise Keras cannot process the data
X_train_real = X_train_real.astype("float32")
X_train_imag = X_train_imag.astype("float32")
y_train = y_train.astype("float32")

X_test_real = X_test_real.astype("float32")
X_test_imag = X_test_imag.astype("float32")
y_test = y_test.astype("float32")

import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

kernels = 32
stride = 1


with strategy.scope():
    
    def residual_block(X, kernels, stride):
        out = tf.keras.layers.BatchNormalization()(X)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv1D(kernels, 3, stride, padding='same')(X)
    
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv1D(kernels, 3, stride, padding='same')(out)
        out = tf.keras.layers.add([X, out])
    ##    out = keras.layers.Conv1D(kernels, stride, padding='same', activation='relu')(out)
    #    out = keras.layers.MaxPool1D(2)(out)
        return out
    
        
    input1 = tf.keras.layers.Input([410,3])
    X = tf.keras.layers.Conv1D(kernels, stride)(input1)
    X = residual_block(X, kernels, stride)
    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)
#    X = tf.keras.layers.Dropout(rate=0.1)(X)
    X = residual_block(X, kernels, stride)
    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)
#    X = tf.keras.layers.Dropout(rate=0.1)(X)
    X = residual_block(X, kernels, stride)
    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)
#    X = tf.keras.layers.Dropout(rate=0.1)(X)
#    X = residual_block(X, kernels, stride)
#    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=3, padding='same')(X)
#    X = tf.keras.layers.Dropout(rate=0.1)(X)
#    X = residual_block(X, kernels, stride)
#    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)

    flat1 = tf.keras.layers.Flatten()(X)

    input2 = tf.keras.layers.Input([410,3])
    X = tf.keras.layers.Conv1D(kernels, stride)(input2)
    X = residual_block(X, kernels, stride)
    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)
 #   X = tf.keras.layers.Dropout(rate=0.1)(X)
    X = residual_block(X, kernels, stride)
    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)
 #   X = tf.keras.layers.Dropout(rate=0.1)(X)
    X = residual_block(X, kernels, stride)
    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)
 #   X = tf.keras.layers.Dropout(rate=0.1)(X)
#    X = residual_block(X, kernels, stride)
#    X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=3, padding='same')(X)
 #   X = tf.keras.layers.Dropout(rate=0.1)(X)
 #   X = residual_block(X, kernels, stride)
 #   X = tf.keras.layers.Conv1D(kernels, kernel_size=5, strides=4, padding='same')(X)

    flat2 = tf.keras.layers.Flatten()(X)

    # merge input models
    merge = tf.keras.layers.concatenate([flat1, flat2])

    # Define the trainable distribution
    def make_masked_autoregressive_flow(index, hidden_units, activation, conditional_event_shape):
    
        made = tfp.bijectors.AutoregressiveNetwork(params=2,
              hidden_units=hidden_units,
              event_shape=(2,),
              activation=activation,
              conditional=True,
              kernel_initializer=tf.keras.initializers.VarianceScaling(0.1),
              conditional_event_shape=conditional_event_shape)
    
        return tfp.bijectors.MaskedAutoregressiveFlow(shift_and_log_scale_fn = made, name='maf'+str(index))

    def make_bijector_kwargs(bijector, name_to_kwargs):
        
        if hasattr(bijector, 'bijectors'):
            
            return {b.name: make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    
        else:
            for name_regex, kwargs in name_to_kwargs.items():
                if re.match(name_regex, bijector.name):
                    return kwargs
        return {}

        
    #input1 = tf.keras.layers.Input([410,2])
    #input2 = tf.keras.layers.Input([410,2])
    x_ = tf.keras.layers.Input(shape=y_train.shape[-1], dtype=tf.float32)
        
    # Define a more expressive model
    num_bijectors = 3
    bijectors = []

    for i in range(num_bijectors):
        masked_auto_i = make_masked_autoregressive_flow(i, hidden_units = [128, 128, 128, 128], activation = 'relu',
                                        conditional_event_shape=merge.shape[-1])
        bijectors.append(masked_auto_i)
    
        USE_BATCHNORM = True
    
        if USE_BATCHNORM and i % 2 == 0:
        # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
            bijectors.append(tfb.BatchNormalization(name='batch_normalization'+str(i)))
    
        bijectors.append(tfb.Permute(permutation = [1, 0]))
        flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
            
    # Define the trainable distribution
    trainable_distribution = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(2)),
                bijector = flow_bijector)

    log_prob_ = trainable_distribution.log_prob(x_, bijector_kwargs=
                make_bijector_kwargs(trainable_distribution.bijector, 
                                     {'maf.': {'conditional_input':merge}}))

    model = tf.keras.Model([input1, input2, x_], log_prob_)
    encoder = tf.keras.Model([input1, input2], merge)  
  
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
    
    # load best model with min validation loss
#    checkpoint.restore('NSBH_3_det/tmp_0xdcd02bf4/ckpt-1')
#    encoder.load_weights("/fred/oz016/Chayan/MAF/GW_localization_MAF/NF_encoder_3_det.hdf5")

model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), loss=lambda _, log_prob: -log_prob)
 
#model,dist = Normalizing_Flows_model().make_distribution(410,2)

class CustomCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, filepath, encoder):
        self.monitor = 'val_loss'
        self.monitor_op = np.less
        self.best = np.Inf

        self.filepath = filepath
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            # self.encoder.save_weights(self.filepath, overwrite=True)
            self.encoder.save(self.filepath, overwrite=True) # Whichever you prefer
        
model.summary()

early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
#checkpoint_cb = keras.callbacks.ModelCheckpoint("Models/ResNet_MDN.h5", save_best_only=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15)

custom_checkpoint = CustomCheckpoint(filepath='/fred/oz016/Chayan/MAF/GW_localization_MAF/NF_encoder_3_det.hdf5',encoder=encoder)

# initialize checkpoints
dataset_name = "NSBH_3_det"
checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

#opt = tf.keras.optimizers.Adam(learning_rate=1e-5)  # optimizer
#checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)

callbacks_list=[custom_checkpoint]  

batch_size = 2000

model.fit([X_train_real, X_train_imag, y_train], np.zeros((len(X_train_real), 0), dtype=np.float32),
          batch_size=batch_size,
          epochs=150,
          validation_split=0.05,
          callbacks=callbacks_list,
          shuffle=True,
          verbose=True)

checkpoint.save(file_prefix=checkpoint_prefix)

encoder.load_weights("/fred/oz016/Chayan/MAF/GW_localization_MAF/NF_encoder_3_det.hdf5")


from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

def kde2D(x, y, bandwidth, ra_pix, de_pix, xbins=150j, ybins=150j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
#    xx, yy = np.mgrid[x.min():x.max():xbins, 
#                      y.min():y.max():ybins]

    xy_sample = np.vstack([de_pix, ra_pix]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(kernel='gaussian', bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)
#    gm = GaussianMixture(n_components=10, random_state=0).fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return z

nside=32
npix=hp.nside2npix(nside)
theta,phi = hp.pixelfunc.pix2ang(nside,np.arange(npix))

# ra_pix and de_pix are co-ordinates in the skuy where I want to find the probabilities
ra_pix = phi
de_pix = -theta + np.pi/2.0 

probs = []
n_samples = 5000

for i in range(len(ra_test_new)):
    x_test_real = np.expand_dims(X_test_real[i],axis=0)
    x_test_imag = np.expand_dims(X_test_imag[i],axis=0)
    
    preds = encoder.predict([x_test_real, x_test_imag])
    
    samples = trainable_distribution.sample((n_samples,),
      bijector_kwargs=make_bijector_kwargs(trainable_distribution.bijector, {'maf.': {'conditional_input':preds}}))
    
    samples = sc.inverse_transform(samples)
    
    ra_samples = samples[:,0]
    dec_samples = samples[:,1]
    
    # A 2D Kernel Density Estimator is used to find the probability density at ra_pix and de_pix
    zz = kde2D(ra_samples,dec_samples, 0.03, ra_pix,de_pix)
    
#    zz[np.where(zz<10**-6)] = 0.0
    zz = zz/(np.sum(zz))

    probs.append(zz)
    
#dec_test_new = np.pi/2.0 - dec_test_new
    
f1 = h5py.File('/fred/oz016/Chayan/SNR_time_series_sample_files/Injection_run_SNR_time_series_NSBH_NF_3_det_new_model.hdf', 'w')
f1.create_dataset('Probabilities', data = probs)
f1.create_dataset('RA_test', data = ra_test_new)
f1.create_dataset('Dec_test', data = dec_test_new)

f1.close()

