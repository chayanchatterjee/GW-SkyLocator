import pandas as pd
import healpy as hp
import numpy as np
import ligo.skymap
import argparse
import matplotlib.pyplot as plt

from matplotlib import rcParams
from pathlib import Path
from ligo.skymap import io, kde, postprocess
from ligo.skymap.plot.marker import reticle

import h5py

##f1 = h5py.File("/fred/oz016/Chayan/GW-SkyNet/evaluation/Bayestar_comparison/Injection_run_BNS_3_det_design_test_0_sec_SNR-8-40_NSIDE-64_ResNet-2D_Bayestar_test_2_0.hdf", "r")
f1 = h5py.File("/fred/oz016/Chayan/GW-SkyLocator/evaluation/Adaptive_NSIDE/Negative_latency/Bayestar_comparison_post-merger/New/Injection_run_BBH_3_det_design_test_Gaussian_KDE_0.hdf", "r")

probs = f1["Probabilities"][()]
ra_preds = f1["RA_samples"][()]
dec_preds = f1["Dec_samples"][()]
ra_test = f1["RA_test"][()]
dec_test = f1["Dec_test"][()]

f1.close()

# Do this only for the new version of the code:
ra_test = np.squeeze(ra_test.T)
dec_test = np.squeeze(dec_test.T)

ra_preds = np.where(ra_preds > 2.0*np.pi, 2.0*np.pi, ra_preds)
ra_preds = np.where(ra_preds < 0.0, 0.0, ra_preds)

dec_preds = np.where(dec_preds > np.pi/2, np.pi/2, dec_preds)
dec_preds = np.where(dec_preds < -np.pi/2, -np.pi/2, dec_preds)

pts = np.stack([ra_preds, dec_preds], axis=2)

sky_posterior = []

import random

#index = random.sample(range(12000), 2000)

#for i in range(len(ra_test)): #pts.shape[0]
#    sky_posterior.append(kde.Clustered2DSkyKDE(pts[i], trials=1, jobs=20))
    
#hpmap = []

#for i in range(len(ra_test)):
#    hpmap.append(sky_posterior[i].as_healpix())
    
#for i in range(len(ra_test)):
#    io.write_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/BNS/skymaps/Test_'+str(i)+'.fits', hpmap[i], nest=True)
    
skymap = []

cl = []
area_90 = []
area_50 = []
search_area = []

eps = 1e-5


for i in range(len(ra_test)): #pts.shape[0]-1
#    s, metadata = io.fits.read_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/NSBH/skymaps/Gaussian_KDE/Test_3_bij_lr_schedule_'+str(i)+'.fits', nest=None)
#    s, metadata = io.fits.read_sky_map('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/45_secs/Test_3_bij_50_epochs_lr_schedule_'+str(i)+'.fits', nest=None)

    s, metadata = io.fits.read_sky_map('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/BBH/skymaps/Gaussian_KDE/Test_new_BN_'+str(i)+'.fits', nest=None)

    skymap = s
    
    # Convert to probability per square degree
    nside = hp.npix2nside(len(skymap))
    deg2perpix = hp.nside2pixarea(nside, degrees=True)
    probperdeg2 = skymap / deg2perpix
    
    event_ra = ra_test[i]
    event_de = dec_test[i]

    vmax = probperdeg2.max()
    vmin = probperdeg2.min()
    
    confidence_levels = postprocess.find_greedy_credible_levels(skymap)
    cl.append(confidence_levels[hp.ang2pix(nside,-event_de+np.pi/2,event_ra, nest=True)]) 
    
    area_90.append(np.sum(confidence_levels <= 0.9*np.sum(skymap)) * hp.nside2pixarea(nside, degrees=True) + eps)
    area_50.append(np.sum(confidence_levels <= 0.5*np.sum(skymap)) * hp.nside2pixarea(nside, degrees=True) + eps)
    
    search_area.append(np.sum(confidence_levels <= cl[i]*np.sum(skymap)) * hp.nside2pixarea(nside, degrees=True) + eps)  
    
from tabulate import tabulate

table = [["90%",np.min(area_90),np.argmin(area_90)],
         ["50%",np.min(area_50),np.argmin(area_50)],
         ["Searched Area",np.min(search_area),np.argmin(search_area)]]

print(tabulate(table, headers=["Percentage", "Minimum area", "Minimum index"]))
  
plt.hist(np.log10(area_90),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='90%, median='+str(round(np.median(area_90),2)))
plt.hist(np.log10(area_50),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='50%, median='+str(round(np.median(area_50),2)))
plt.hist(np.log10(search_area),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='Search, median='+str(round(np.median(search_area),2)))


#f1 = h5py.File('/fred/oz016/Chayan/GW-SkyNet_pre-merger/evaluation/skymaps/CPU/Pre-merger/New/Injection_run_BNS_3_det_design_test_45_sec_SNR-8-40_Adaptive_Gaussian_KDE_3_bij_50_epochs_lr_sch_bandw_003_stdscale.hdf', 'w')
f1 = h5py.File('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/BBH/Injection_run_BBH_3_det_design_test_Gaussian_KDE_new_BN.hdf', 'w')
f1.create_dataset('Area-90', data=area_90)
f1.create_dataset('Area-50', data=area_50)
f1.create_dataset('Area-Searched', data=search_area)
#f1.create_dataset('Index', data=index)

f1.close()

plt.legend(loc=4)
plt.ylabel('Cumulative Ratio')
plt.xlabel('Area in log(deg^2)')
plt.savefig('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/BBH/Area_ResNet_NF_BBH_3_det_Gaussian_KDE_new_BN.png', dpi=400)


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='pp_plot')

ax.add_confidence_band(len(ra_test),alpha=0.95) # Add 90% confidence band
ax.add_diagonal() # Add diagonal line
#ax.add_lightning(len(ra_test), 20) # Add some random realizations of n samples
ax.add_series(cl) # Add our data
plt.xlabel('Credible intervals')
plt.ylabel('Cumulative distribution')
plt.savefig('/fred/oz016/Chayan/GW-SkyLocator/evaluation/skymaps/CPU/BBH/CLvCR_test_ResNet_NF_BBH_3_det_Gaussian_KDE_new_BN.png', dpi=400)

print('Done!')
