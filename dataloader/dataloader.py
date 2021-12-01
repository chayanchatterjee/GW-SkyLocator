"""Data Loader"""

import h5py

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_train_data(data_config):
        """Loads dataset from path"""
        f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
        f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
        f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
        f4 = h5py.File(data_config.NSBH.path_train_4, 'r')

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
        
        h1_real = h1_real[:,:,None]
        l1_real = l1_real[:,:,None]
        v1_real = v1_real[:,:,None]

        h1_imag = h1_imag[:,:,None]
        l1_imag = l1_imag[:,:,None]
        v1_imag = v1_imag[:,:,None]

        X_train_real = np.concatenate((h1_real, l1_real, v1_real), axis=2)
        X_train_imag = np.concatenate((h1_imag, l1_imag, v1_imag), axis=2)
        
        return X_train_real, X_train_imag
    
    @staticmethod
    def load_train_parameters(data_config):
        """Loads train parameters from path"""
        f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
        f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
        f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
        f4 = h5py.File(data_config.NSBH.path_train_4, 'r')

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
        
        ra = ra[:,None]
        dec = dec[:,None]

        y_train = np.concatenate((ra, dec), axis=1)

        return y_train
    
    @staticmethod
    def load_test_data(data_config):
        """Loads dataset from path"""
        #Get the HDF5 group
        f_test = h5py.File(data_config.NSBH.path_test, 'r')
        group_test = f_test['omf_injection_snr_samples']

        data_h1_test = group_test['h1_snr']
        data_l1_test = group_test['l1_snr']
        data_v1_test = group_test['v1_snr']

        for i in range(data_config.train.num_test):
            h1_test_real[i] = abs(data_h1_test[str(i)][()][1840:2250])
            l1_test_real[i] = abs(data_l1_test[str(i)][()][1840:2250])
            v1_test_real[i] = abs(data_v1_test[str(i)][()][1840:2250])
    
            h1_test_imag[i] = np.imag(data_h1_test[str(i)][()][1840:2250])
            l1_test_imag[i] = np.imag(data_l1_test[str(i)][()][1840:2250])
            v1_test_imag[i] = np.imag(data_v1_test[str(i)][()][1840:2250])
        
        f_test.close()
        
        h1_test_real = h1_test_real[:,:,None]
        l1_test_real = l1_test_real[:,:,None]
        v1_test_real = v1_test_real[:,:,None]

        h1_test_imag = h1_test_imag[:,:,None]
        l1_test_imag = l1_test_imag[:,:,None]
        v1_test_imag = v1_test_imag[:,:,None]

        X_test_real = np.concatenate((h1_test_real, l1_test_real, v1_test_real), axis=2)
        X_test_imag = np.concatenate((h1_test_imag, l1_test_imag, v1_test_imag), axis=2)
       
        return X_test_real, X_test_imag

           
    @staticmethod
    def load_test_parameters(data_config):
        """Loads train parameters from path"""
        f_test = h5py.File(data_config.NSBH.path_test, 'r')
        
        data_ra = f_test['ra'][()]
        data_dec = f_test['dec'][()]

        f_test.close()

        ra_test = 2.0*np.pi*data_ra
        dec_test = np.arcsin(1.0 - 2.0*data_dec)
        
        ra_test_new = ra_test_new[:,None]
        dec_test_new = dec_test_new[:,None]

        y_test = np.concatenate((ra_test_new, dec_test_new), axis=1)

        return y_test
        
        
        

