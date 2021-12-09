import numpy as np
import healpy
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
import ligo.skymap.plot

import h5py

############################################ For Normalizing Flow ##############################################################

f1 = h5py.File('Injection_run_SNR_time_series_NSBH_NF_3_det_new_model_ResNet.hdf', 'r')
p = f1['Probabilities'][()]
ra_test = f1['RA_test'][()]
dec_test = f1['Dec_test'][()]

#p_new = []
#for i in range(2000):
#    p_new.append(p[i]/np.sum(p[i]))

# Do this only for the new version of the code:
ra_test = np.squeeze(ra_test.T)
dec_test = np.squeeze(dec_test.T)

################################################################################################################################

def my_contour_area(array,sig_level):
    ratio = float(sig_level)/100
    arg = np.argsort(-array)
    total = np.sum(array)
    part = 0.0
    k=0
    for i in arg:
        part = part + array[i]
        if part/total>(ratio) and k==0:
            k = k+1    # k = 0 in area calculation results in zero area which leads to error when we take log10
            break
        elif part/total>(ratio) and k!=0:
            break
        k = k+1
    
#    area = (1-float(k)/float(len(array)))*4*np.pi
    area = (float(k)/float(len(array)))*4*np.pi*3282.81
    return area


def my_cl(array,value):
    ratio=1.0
    arg = np.argsort(-array)
    total = np.sum(array)
    part=0.0
    for i in arg:
#        part = part + array[i]
        if array[i]<value:
            ratio = part/total
            break
        else:
            part = part + array[i]
    
    return ratio

#cl = []
#search = []
#area90 = []
#area50 = []
#for i in range(5000):
##    map_gps = np.asarray([str('Healpy_Predictions/Healpy_Preds_')+str(i)+str('.fits')])
#    map_gps = prob[i] # reading probability distribution data
##    n = []
##    area90 = []
##    area50 = []
##    n = healpy.fitsfunc.read_map(map_gps)
##    print(sum(n))
#    declination = dec_test[i]
#    right_ascension = ra_test[i]
#    value = healpy.pixelfunc.get_interp_val(map_gps,-declination+np.pi/2,right_ascension) # getting the contour level at a
#    cl.append(my_cl(map_gps,value)*100)                                                   # particular RA and Dec using
#    area90.append(my_contour_area(map_gps,90))                                            # interpolation.
#    area50.append(my_contour_area(map_gps,50)) # calculating area of the 90% and 50% contours
#    search.append(my_contour_area(map_gps,cl[i])) # calculating area of arbitrary confidence interval

#print(cl)
#print(search)

import healpy as hp
from ligo.skymap import postprocess

cl = []
area_90 = []
area_50 = []
search_area = []
nside = 32

#p = np.stack(p, axis=0)

eps = 1e-5

for i in range(len(ra_test)):
    cls = postprocess.find_greedy_credible_levels(p[i])
    declination = dec_test[i]
    right_ascension = ra_test[i]
    cl.append(cls[hp.ang2pix(nside,-declination+np.pi/2,right_ascension)])   
    
for i in range(len(dec_test)):
    cls = postprocess.find_greedy_credible_levels(p[i])
    area_90.append(np.sum(cls <= 0.9*np.sum(p[i])) * hp.nside2pixarea(nside, degrees=True) + eps)
    area_50.append(np.sum(cls <= 0.5*np.sum(p[i])) * hp.nside2pixarea(nside, degrees=True) + eps)
    
    declination = dec_test[i]
    right_ascension = ra_test[i]
    value = healpy.pixelfunc.get_interp_val(p[i],-declination+np.pi/2,right_ascension)
    
    search_area.append(np.sum(cls <= cl[i]*np.sum(p[i])) * hp.nside2pixarea(nside, degrees=True) + eps)  
    
#print(np.min(area_90))
#print(np.min(area_50))
#print(np.min(search_area))

from tabulate import tabulate

table = [["90%",np.min(area_90),np.argmin(area_90)],
         ["50%",np.min(area_50),np.argmin(area_50)],
         ["Searched Area",np.min(search_area),np.argmin(search_area)]]

print(tabulate(table, headers=["Percentage", "Minimum area", "Minimum index"]))
    
#x_arr = np.array([0,100])
#y_arr = np.array([0,1])
#plt.hist(cl,20,cumulative=True,histtype='step',normed=True)
#plt.plot(x_arr,y_arr,'--')
#plt.xlim(0,100)
#plt.xlabel('Confidence Level')
#plt.ylabel('Cumulative Ratio')
#plt.savefig('Injection_run_probabilistic_classification/CLvCR_test_Gaussian_3_det_classification_KDE_BNS_new.png', dpi=400) # Plotting the P-P plot

#plt.clf()
#plt.show()

# Getting the histograms of the 50%, 90% and searched areas

plt.hist(np.log10(area_90),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='90%, median='+str(round(np.median(area_90),2)))
plt.hist(np.log10(area_50),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='50%, median='+str(round(np.median(area_50),2)))
plt.hist(np.log10(search_area),50,range=(1,np.max(np.log10(search_area))),cumulative=True,histtype='step',density=True,label='Search, median='+str(round(np.median(search_area),2)))


plt.legend(loc=4)
plt.ylabel('Cumulative Ratio')
plt.xlabel('Area in log(deg^2)')
plt.savefig('Plots/Area_WaveNet_NF_NSBH_3_det_train_NSBH_ResNet-34.png', dpi=400)


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='pp_plot')

ax.add_confidence_band(len(ra_test),alpha=0.90) # Add 90% confidence band
ax.add_diagonal() # Add diagonal line
#ax.add_lightning(len(ra_test), 20) # Add some random realizations of n samples
ax.add_series(cl) # Add our data
plt.savefig('Plots/CLvCR_test_WaveNet_NF_NSBH_3_det_train_NSBH_ResNet-34.png', dpi=400)

