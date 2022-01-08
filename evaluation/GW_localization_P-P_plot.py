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

f1 = h5py.File("Fixed_NSIDE/Injection_run_BNS_new_ResNet-34.hdf", "r")
probs = f1["Probabilities"][()]
ra_preds = f1["RA_samples"][()]
dec_preds = f1["Dec_samples"][()]
ra_test = f1["RA_test"][()]
dec_test = f1["Dec_test"][()]

f1.close()

ra_preds = np.where(ra_preds > 2.0*np.pi, 2.0*np.pi, ra_preds)
ra_preds = np.where(ra_preds < 0.0, 0.0, ra_preds)

dec_preds = np.where(dec_preds > np.pi/2, np.pi/2, dec_preds)
dec_preds = np.where(dec_preds < -np.pi/2, -np.pi/2, dec_preds)

pts = np.stack([ra_preds, dec_preds], axis=2)

sky_posterior = []

for i in range(pts.shape[0]):
    sky_posterior.append(kde.Clustered2DSkyKDE(pts[i], trials=1, jobs=25))
    
hpmap = []

for i in range(pts.shape[0]):
    hpmap.append(sky_posterior[i].as_healpix())
    
for i in range(pts.shape[0]):
    io.write_sky_map('/group/pmc005/cchatterjee/skymaps/Test_'+str(i)+'.fits', hpmap[i], nest=True)
    
skymap = []

for i in range(pts.shape[0]):
    s, metadata = io.fits.read_sky_map('/group/pmc005/cchatterjee/skymaps/Test_'+str(i)+'.fits', nest=None)
    skymap.append(s)
    
ax = plt.axes(projection="astro hours mollweide")
ax.grid()

cl = []
area_90 = []
area_50 = []
search_area = []

eps = 1e-5

for i in range(pts.shape[0]):
    # Convert to probability per square degree
    nside = hp.npix2nside(len(skymap[i]))
    deg2perpix = hp.nside2pixarea(nside, degrees=True)
    probperdeg2 = skymap[i] / deg2perpix
    
    event_ra = ra_test[i]
    event_de = dec_test[i]

    vmax = probperdeg2.max()
    vmin = probperdeg2.min()
    
    confidence_levels = 100 * postprocess.find_greedy_credible_levels(skymap[i])
    cl.append(confidence_levels[hp.ang2pix(nside,-event_de+np.pi/2,event_ra, nest=True)]) 
    
    area_90.append(np.sum(confidence_levels <= 90) * hp.nside2pixarea(nside, degrees=True) + eps)
    area_50.append(np.sum(confidence_levels <= 50) * hp.nside2pixarea(nside, degrees=True) + eps)
    
    search_area.append(np.sum(confidence_levels <= cl[i]) * hp.nside2pixarea(nside, degrees=True) + eps)  
    
from tabulate import tabulate

table = [["90%",np.min(area_90),np.argmin(area_90)],
         ["50%",np.min(area_50),np.argmin(area_50)],
         ["Searched Area",np.min(search_area),np.argmin(search_area)]]

print(tabulate(table, headers=["Percentage", "Minimum area", "Minimum index"]))
  
plt.hist(np.log10(area_90),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='90%, median='+str(round(np.median(area_90),2)))
plt.hist(np.log10(area_50),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='50%, median='+str(round(np.median(area_50),2)))
plt.hist(np.log10(search_area),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='Search, median='+str(round(np.median(search_area),2)))


plt.legend(loc=4)
plt.ylabel('Cumulative Ratio')
plt.xlabel('Area in log(deg^2)')
plt.savefig('Plots/Adaptive_Refinement/Area_ResNet_NF_BNS_3_det_train.png', dpi=400)


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='pp_plot')

ax.add_confidence_band(len(ra_test),alpha=0.90) # Add 90% confidence band
ax.add_diagonal() # Add diagonal line
#ax.add_lightning(len(ra_test), 20) # Add some random realizations of n samples
ax.add_series(cl) # Add our data
plt.savefig('Plots/Adaptive_Refinement/CLvCR_test_ResNet_NF_BNS_3_det.png', dpi=400)
