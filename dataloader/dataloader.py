"""Data Loader"""

# Imports
import numpy as np
import h5py

class DataLoader:
    """Data Loader class"""
    def __init__(self, num_det, dataset, n_test, n_samples, min_snr, train_negative_latency, train_negative_latency_seconds):
        
        self.num_det = num_det
        self.dataset = dataset
        self.n_test = n_test
        self.n_samples = n_samples
        self.min_snr = min_snr
        self.train_negative_latency = train_negative_latency
        self.train_negative_latency_seconds = train_negative_latency_seconds
        
    
#    @staticmethod
    def load_train_3_det_data(self, data_config, snr_range_train, train_real):
        """Loads dataset from path"""
        # NSBH
        if((self.dataset == 'NSBH') and (snr_range_train=='high')):
            f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
            f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
            f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
            f4 = h5py.File(data_config.NSBH.path_train_4, 'r')
            
            h1_real_52k = abs(f1['h1_snr_series'][()] )
            l1_real_52k = abs(f1['l1_snr_series'][()] )
            v1_real_52k = abs(f1['v1_snr_series'][()] )
        
            h1_real_30k = abs(f2['h1_snr_series'][()] )
            l1_real_30k = abs(f2['l1_snr_series'][()] )
            v1_real_30k = abs(f2['v1_snr_series'][()] )
            
            h1_real_12k = abs(f3['h1_snr_series'][()] )
            l1_real_12k = abs(f3['l1_snr_series'][()] )
            v1_real_12k = abs(f3['v1_snr_series'][()] )
        
            h1_real_6k = abs(f4['h1_snr_series'][()] )
            l1_real_6k = abs(f4['l1_snr_series'][()] )
            v1_real_6k = abs(f4['v1_snr_series'][()] )
            
            h1_real = np.concatenate([h1_real_52k, h1_real_30k, h1_real_12k, h1_real_6k], axis=0)
            l1_real = np.concatenate([l1_real_52k, l1_real_30k, l1_real_12k, l1_real_6k], axis=0)
            v1_real = np.concatenate([v1_real_52k, v1_real_30k, v1_real_12k, v1_real_6k], axis=0)
            
            h1_imag_52k = np.imag(f1['h1_snr_series'][()] )
            l1_imag_52k = np.imag(f1['l1_snr_series'][()] )
            v1_imag_52k = np.imag(f1['v1_snr_series'][()] )
        
            h1_imag_30k = np.imag(f2['h1_snr_series'][()] )
            l1_imag_30k = np.imag(f2['l1_snr_series'][()] )
            v1_imag_30k = np.imag(f2['v1_snr_series'][()] )
        
            h1_imag_12k = np.imag(f3['h1_snr_series'][()] )
            l1_imag_12k = np.imag(f3['l1_snr_series'][()] )
            v1_imag_12k = np.imag(f3['v1_snr_series'][()] )
        
            h1_imag_6k = np.imag(f4['h1_snr_series'][()] )
            l1_imag_6k = np.imag(f4['l1_snr_series'][()] )
            v1_imag_6k = np.imag(f4['v1_snr_series'][()] )
            
            h1_imag = np.concatenate([h1_imag_52k, h1_imag_30k, h1_imag_12k, h1_imag_6k], axis=0)
            l1_imag = np.concatenate([l1_imag_52k, l1_imag_30k, l1_imag_12k, l1_imag_6k], axis=0)
            v1_imag = np.concatenate([v1_imag_52k, v1_imag_30k, v1_imag_12k, v1_imag_6k], axis=0)
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            
        elif((self.dataset == 'NSBH') and (snr_range_train=='low')):
            
            f1 = h5py.File(data_config.NSBH.path_train_low_snr_1, 'r')
            f2 = h5py.File(data_config.NSBH.path_train_low_snr_2, 'r')

            h1_real_60k = abs(f1['h1_snr_series'][0:60000][()] )
            l1_real_60k = abs(f1['l1_snr_series'][0:60000][()] )
            v1_real_60k = abs(f1['v1_snr_series'][0:60000][()] )

            h1_imag_60k = abs(np.imag(f1['h1_snr_series'][0:60000][()]))
            l1_imag_60k = abs(np.imag(f1['l1_snr_series'][0:60000][()]))
            v1_imag_60k = abs(np.imag(f1['v1_snr_series'][0:60000][()]))
            
            h1_real_40k = abs(f2['h1_snr_series'][0:40000][()] )
            l1_real_40k = abs(f2['l1_snr_series'][0:40000][()] )
            v1_real_40k = abs(f2['v1_snr_series'][0:40000][()] )
    
            h1_imag_40k = abs(np.imag(f2['h1_snr_series'][0:40000][()]))
            l1_imag_40k = abs(np.imag(f2['l1_snr_series'][0:40000][()]))
            v1_imag_40k = abs(np.imag(f2['v1_snr_series'][0:40000][()]))
                
            h1_real = np.concatenate([h1_real_60k, h1_real_40k], axis=0)
            l1_real = np.concatenate([l1_real_60k, l1_real_40k], axis=0)
            v1_real = np.concatenate([v1_real_60k, v1_real_40k], axis=0)
            
            h1_imag = np.concatenate([h1_imag_60k, h1_imag_40k], axis=0)
            l1_imag = np.concatenate([l1_imag_60k, l1_imag_40k], axis=0)
            v1_imag = np.concatenate([v1_imag_60k, v1_imag_40k], axis=0)
            
            f1.close()
            f2.close()
            
        # BBH
        elif((self.dataset == 'BBH') and (snr_range_train=='high')):
            f1 = h5py.File(data_config.BBH.path_train, 'r')

            h1_real = abs(f1['h1_snr_series'][0:100000][()] )
            l1_real = abs(f1['l1_snr_series'][0:100000][()] )
            v1_real = abs(f1['v1_snr_series'][0:100000][()] )

            h1_imag = np.imag(f1['h1_snr_series'][0:100000][()] )
            l1_imag = np.imag(f1['l1_snr_series'][0:100000][()] )
            v1_imag = np.imag(f1['v1_snr_series'][0:100000][()] )
    
            f1.close()
        
        elif((self.dataset == 'BBH') and (snr_range_train=='low')):
            
            f1 = h5py.File(data_config.BBH.path_train_low_SNR, 'r')

            h1_real = abs(f1['h1_snr_series'][0:100000][()] )
            l1_real = abs(f1['l1_snr_series'][0:100000][()] )
            v1_real = abs(f1['v1_snr_series'][0:100000][()] )

            h1_imag = np.imag(f1['h1_snr_series'][0:100000][()] )
            l1_imag = np.imag(f1['l1_snr_series'][0:100000][()] )
            v1_imag = np.imag(f1['v1_snr_series'][0:100000][()] )
    
            f1.close()
        
        # BNS
        elif(self.dataset == 'BNS'):
            if((snr_range_train == 'high') and (train_real == False) and (self.train_negative_latency == False)):
                f1 = h5py.File(data_config.BNS.path_train_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_2, 'r')

                h1_real_22k = abs(f1['h1_snr_series'][0:22000][()] )
                l1_real_22k = abs(f1['l1_snr_series'][0:22000][()] )
                v1_real_22k = abs(f1['v1_snr_series'][0:22000][()] )

                h1_imag_22k = abs(np.imag(f1['h1_snr_series'][0:22000][()]))
                l1_imag_22k = abs(np.imag(f1['l1_snr_series'][0:22000][()]))
                v1_imag_22k = abs(np.imag(f1['v1_snr_series'][0:22000][()]))
            
                h1_real_86k = abs(f2['h1_snr_series'][0:86000][()] )
                l1_real_86k = abs(f2['l1_snr_series'][0:86000][()] )
                v1_real_86k = abs(f2['v1_snr_series'][0:86000][()] )
    
                h1_imag_86k = abs(np.imag(f2['h1_snr_series'][0:86000][()]))
                l1_imag_86k = abs(np.imag(f2['l1_snr_series'][0:86000][()]))
                v1_imag_86k = abs(np.imag(f2['v1_snr_series'][0:86000][()]))
            
                h1_real = np.concatenate([h1_real_22k, h1_real_86k], axis=0)
                l1_real = np.concatenate([l1_real_22k, l1_real_86k], axis=0)
                v1_real = np.concatenate([v1_real_22k, v1_real_86k], axis=0)
            
                h1_imag = np.concatenate([h1_imag_22k, h1_imag_86k], axis=0)
                l1_imag = np.concatenate([l1_imag_22k, l1_imag_86k], axis=0)
                v1_imag = np.concatenate([v1_imag_22k, v1_imag_86k], axis=0)
            
                f1.close()
                f2.close()
                
            elif((snr_range_train == 'high') and (train_real == True) and (self.train_negative_latency == False)):
#                f1 = h5py.File(data_config.BNS.path_train_real_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_real_2, 'r')
                f3 = h5py.File(data_config.BNS.path_train_real_3, 'r')

#                h1_real_12k = abs(f1['h1_snr_series'][()] )
#                l1_real_12k = abs(f1['l1_snr_series'][()] )
#                v1_real_12k = abs(f1['v1_snr_series'][()] )

#                h1_imag_12k = abs(np.imag(f1['h1_snr_series'][()]))
#                l1_imag_12k = abs(np.imag(f1['l1_snr_series'][()]))
#                v1_imag_12k = abs(np.imag(f1['v1_snr_series'][()]))
            
                h1_real_42k = abs(f2['h1_snr_series'][0:46000][()] )
                l1_real_42k = abs(f2['l1_snr_series'][0:46000][()] )
                v1_real_42k = abs(f2['v1_snr_series'][0:46000][()] )
    
                h1_imag_42k = abs(np.imag(f2['h1_snr_series'][0:46000][()]))
                l1_imag_42k = abs(np.imag(f2['l1_snr_series'][0:46000][()]))
                v1_imag_42k = abs(np.imag(f2['v1_snr_series'][0:46000][()]))
                
                h1_real_46k = abs(f3['h1_snr_series'][0:42000][()] )
                l1_real_46k = abs(f3['l1_snr_series'][0:42000][()] )
                v1_real_46k = abs(f3['v1_snr_series'][0:42000][()] )
    
                h1_imag_46k = abs(np.imag(f3['h1_snr_series'][0:42000][()]))
                l1_imag_46k = abs(np.imag(f3['l1_snr_series'][0:42000][()]))
                v1_imag_46k = abs(np.imag(f3['v1_snr_series'][0:42000][()]))
            
                h1_real = np.concatenate([h1_real_42k, h1_real_46k], axis=0)
                l1_real = np.concatenate([l1_real_42k, l1_real_46k], axis=0)
                v1_real = np.concatenate([v1_real_42k, v1_real_46k], axis=0)
            
                h1_imag = np.concatenate([h1_imag_42k, h1_imag_46k], axis=0)
                l1_imag = np.concatenate([l1_imag_42k, l1_imag_46k], axis=0)
                v1_imag = np.concatenate([v1_imag_42k, v1_imag_46k], axis=0)
            
#                f1.close()
                f2.close()
                f3.close()
                
            elif((snr_range_train == 'low') and (train_real == False) and (self.train_negative_latency == False)):
                
                f1 = h5py.File(data_config.BNS.path_train_low_snr_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_low_snr_2, 'r')
                f3 = h5py.File(data_config.BNS.path_train_low_snr_3, 'r')

                h1_real_12k = abs(f1['h1_snr_series'][0:12000][()] )
                l1_real_12k = abs(f1['l1_snr_series'][0:12000][()] )
                v1_real_12k = abs(f1['v1_snr_series'][0:12000][()] )

                h1_imag_12k = np.imag(f1['h1_snr_series'][0:12000][()] )
                l1_imag_12k = np.imag(f1['l1_snr_series'][0:12000][()] )
                v1_imag_12k = np.imag(f1['v1_snr_series'][0:12000][()] )
            
                h1_real_36k = abs(f2['h1_snr_series'][0:36000][()] )
                l1_real_36k = abs(f2['l1_snr_series'][0:36000][()] )
                v1_real_36k = abs(f2['v1_snr_series'][0:36000][()] )
    
                h1_imag_36k = np.imag(f2['h1_snr_series'][0:36000][()] )
                l1_imag_36k = np.imag(f2['l1_snr_series'][0:36000][()] )
                v1_imag_36k = np.imag(f2['v1_snr_series'][0:36000][()] )
                
                h1_real_52k = abs(f3['h1_snr_series'][0:52000][()] )
                l1_real_52k = abs(f3['l1_snr_series'][0:52000][()] )
                v1_real_52k = abs(f3['v1_snr_series'][0:52000][()] )
    
                h1_imag_52k = np.imag(f3['h1_snr_series'][0:52000][()] )
                l1_imag_52k = np.imag(f3['l1_snr_series'][0:52000][()] )
                v1_imag_52k = np.imag(f3['v1_snr_series'][0:52000][()] )
            
                h1_real = np.concatenate([h1_real_12k, h1_real_36k, h1_real_52k], axis=0)
                l1_real = np.concatenate([l1_real_12k, l1_real_36k, l1_real_52k], axis=0)
                v1_real = np.concatenate([v1_real_12k, v1_real_36k, v1_real_52k], axis=0)
            
                h1_imag = np.concatenate([h1_imag_12k, h1_imag_36k, h1_imag_52k], axis=0)
                l1_imag = np.concatenate([l1_imag_12k, l1_imag_36k, l1_imag_52k], axis=0)
                v1_imag = np.concatenate([v1_imag_12k, v1_imag_36k, v1_imag_52k], axis=0)
            
                f1.close()
                f2.close()
                f3.close()
        
            
            elif((self.train_negative_latency == True) and (self.train_negative_latency_seconds == '5')):
                
                f1 = h5py.File(data_config.BNS.path_train_5_sec, 'r')

                h1_real = abs(f1['h1_snr_series'][()] )
                l1_real = abs(f1['l1_snr_series'][()] )
                v1_real = abs(f1['v1_snr_series'][()] )

                h1_imag = abs(np.imag(f1['h1_snr_series'][()]))
                l1_imag = abs(np.imag(f1['l1_snr_series'][()]))
                v1_imag = abs(np.imag(f1['v1_snr_series'][()]))
            
                f1.close()
                
            elif((self.train_negative_latency == True) and (self.train_negative_latency_seconds == '10')):
                
                f1 = h5py.File(data_config.BNS.path_train_10_sec, 'r')

                h1_real = abs(f1['h1_snr_series'][()] )
                l1_real = abs(f1['l1_snr_series'][()] )
                v1_real = abs(f1['v1_snr_series'][()] )

                h1_imag = abs(np.imag(f1['h1_snr_series'][()]))
                l1_imag = abs(np.imag(f1['l1_snr_series'][()]))
                v1_imag = abs(np.imag(f1['v1_snr_series'][()]))
            
                f1.close()
                
                
        h1_real = h1_real[:,:,None]
        l1_real = l1_real[:,:,None]
        v1_real = v1_real[:,:,None]

        h1_imag = h1_imag[:,:,None]
        l1_imag = l1_imag[:,:,None]
        v1_imag = v1_imag[:,:,None]
        
        X_train_real = np.concatenate((h1_real, l1_real, v1_real), axis=2)
        X_train_imag = np.concatenate((h1_imag, l1_imag, v1_imag), axis=2)
        
        return X_train_real, X_train_imag
    
    def load_train_2_det_data(self, data_config, snr_range_train):
        """Loads dataset from path"""
        # BNS dataset
        if(self.train_negative_latency == False):
            if((self.dataset == 'BNS') and (snr_range_train == 'low')):
                f1 = h5py.File(data_config.BNS.path_train_2_det_low_SNR, 'r')
                f2 = h5py.File(data_config.BNS.path_train_low_snr_1, 'r')
                f3 = h5py.File(data_config.BNS.path_train_low_snr_2, 'r')
                f4 = h5py.File(data_config.BNS.path_train_low_snr_3, 'r')  
                f5 = h5py.File(data_config.BNS.path_train_1, 'r')
                f6 = h5py.File(data_config.BNS.path_train_2, 'r')
                f7 = h5py.File(data_config.BNS.path_train_2_det_low_SNR_1, 'r')
                f8 = h5py.File(data_config.BNS.path_train_2_det_high_SNR_1, 'r')
                f9 = h5py.File(data_config.BNS.path_train_2_det_high_SNR_2, 'r')
            
                h1_real = abs(f1['h1_snr_series'][()])
                l1_real = abs(f1['l1_snr_series'][()])
            
                h1_imag = abs(np.imag(f1['h1_snr_series'][()]))
                l1_imag = abs(np.imag(f1['l1_snr_series'][()]))

                h1_real_12k = abs(f2['h1_snr_series'][0:12000][()])
                l1_real_12k = abs(f2['l1_snr_series'][0:12000][()])

                h1_imag_12k = abs(np.imag(f2['h1_snr_series'][0:12000][()]))
                l1_imag_12k = abs(np.imag(f2['l1_snr_series'][0:12000][()]))
            
                h1_real_36k = abs(f3['h1_snr_series'][0:36000][()] )
                l1_real_36k = abs(f3['l1_snr_series'][0:36000][()] )
    
                h1_imag_36k = abs(np.imag(f3['h1_snr_series'][0:36000][()]))
                l1_imag_36k = abs(np.imag(f3['l1_snr_series'][0:36000][()]))
                
                h1_real_52k = abs(f4['h1_snr_series'][0:52000][()] )
                l1_real_52k = abs(f4['l1_snr_series'][0:52000][()] )
              
                h1_imag_52k = abs(np.imag(f4['h1_snr_series'][0:52000][()]))
                l1_imag_52k = abs(np.imag(f4['l1_snr_series'][0:52000][()]))
            
                h1_real_22k = abs(f5['h1_snr_series'][()])
                l1_real_22k = abs(f5['l1_snr_series'][()])
    
                h1_imag_22k = abs(np.imag(f5['h1_snr_series'][()]))
                l1_imag_22k = abs(np.imag(f5['l1_snr_series'][()]))
            
                h1_real_86k = abs(f6['h1_snr_series'][()])
                l1_real_86k = abs(f6['l1_snr_series'][()])
    
                h1_imag_86k = abs(np.imag(f6['h1_snr_series'][()]))
                l1_imag_86k = abs(np.imag(f6['l1_snr_series'][()]))
            
                h1_real_102k = abs(f7['h1_snr_series'][()])
                l1_real_102k = abs(f7['l1_snr_series'][()])
    
                h1_imag_102k = abs(np.imag(f7['h1_snr_series'][()]))
                l1_imag_102k = abs(np.imag(f7['l1_snr_series'][()]))
            
                h1_real_high_1 = abs(f8['h1_snr_series'][()])
                l1_real_high_1 = abs(f8['l1_snr_series'][()])
            
                h1_imag_high_1 = abs(np.imag(f8['h1_snr_series'][()]))
                l1_imag_high_1 = abs(np.imag(f8['l1_snr_series'][()]))
            
                h1_real_high_2 = abs(f9['h1_snr_series'][()])
                l1_real_high_2 = abs(f9['l1_snr_series'][()])
            
                h1_imag_high_2 = abs(np.imag(f9['h1_snr_series'][()]))
                l1_imag_high_2 = abs(np.imag(f9['l1_snr_series'][()]))
            
                        
                h1_real = np.concatenate([h1_real, h1_real_12k, h1_real_36k, h1_real_52k, h1_real_22k, h1_real_86k, h1_real_102k], axis=0)
                l1_real = np.concatenate([l1_real, l1_real_12k, l1_real_36k, l1_real_52k, l1_real_22k, l1_real_86k, l1_real_102k], axis=0)
            
                h1_imag = np.concatenate([h1_imag, h1_imag_12k, h1_imag_36k, h1_imag_52k, h1_imag_22k, h1_imag_86k, h1_imag_102k], axis=0)
                l1_imag = np.concatenate([l1_imag, l1_imag_12k, l1_imag_36k, l1_imag_52k, l1_imag_22k, l1_imag_86k, l1_imag_102k], axis=0)
            
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
                f8.close()
                f9.close()
            
            # NSBH dataset
            elif((self.dataset == 'NSBH') and (snr_range_train == 'low')):
                f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
                f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
                f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
                f4 = h5py.File(data_config.NSBH.path_train_4, 'r')
                f5 = h5py.File(data_config.NSBH.path_train_low_snr_1, 'r')
                f6 = h5py.File(data_config.NSBH.path_train_low_snr_2, 'r')
                f7 = h5py.File(data_config.NSBH.path_train_low_snr_3, 'r')
            
                h1_real_52k = abs(f1['h1_snr_series'][()])
                l1_real_52k = abs(f1['l1_snr_series'][()])
        
                h1_real_30k = abs(f2['h1_snr_series'][()])
                l1_real_30k = abs(f2['l1_snr_series'][()])
            
                h1_real_12k = abs(f3['h1_snr_series'][()])
                l1_real_12k = abs(f3['l1_snr_series'][()])
        
                h1_real_6k = abs(f4['h1_snr_series'][()])
                l1_real_6k = abs(f4['l1_snr_series'][()])
            
                h1_real_60k = abs(f5['h1_snr_series'][0:60000][()] )
                l1_real_60k = abs(f5['l1_snr_series'][0:60000][()] )
            
                h1_real_40k = abs(f6['h1_snr_series'][0:40000][()] )
                l1_real_40k = abs(f6['l1_snr_series'][0:40000][()] )
            
                h1_real_72k = abs(f7['h1_snr_series'][()] )
                l1_real_72k = abs(f7['l1_snr_series'][()] )
                
                h1_imag_52k = abs(np.imag(f1['h1_snr_series'][()]))
                l1_imag_52k = abs(np.imag(f1['l1_snr_series'][()]))
        
                h1_imag_30k = abs(np.imag(f2['h1_snr_series'][()]))
                l1_imag_30k = abs(np.imag(f2['l1_snr_series'][()]))
        
                h1_imag_12k = abs(np.imag(f3['h1_snr_series'][()]))
                l1_imag_12k = abs(np.imag(f3['l1_snr_series'][()]))
        
                h1_imag_6k = abs(np.imag(f4['h1_snr_series'][()]))
                l1_imag_6k = abs(np.imag(f4['l1_snr_series'][()]))
            
                h1_imag_60k = abs(np.imag(f5['h1_snr_series'][0:60000][()]))
                l1_imag_60k = abs(np.imag(f5['l1_snr_series'][0:60000][()]))
            
                h1_imag_40k = abs(np.imag(f6['h1_snr_series'][0:40000][()]))
                l1_imag_40k = abs(np.imag(f6['l1_snr_series'][0:40000][()]))
            
                h1_imag_72k = abs(np.imag(f7['h1_snr_series'][()]))
                l1_imag_72k = abs(np.imag(f7['l1_snr_series'][()]))
            
            
                h1_real = np.concatenate([h1_real_52k, h1_real_30k, h1_real_12k, h1_real_6k, h1_real_60k, h1_real_40k, h1_real_72k], axis=0)
                l1_real = np.concatenate([l1_real_52k, l1_real_30k, l1_real_12k, l1_real_6k, l1_real_60k, l1_real_40k, l1_real_72k], axis=0)
            
            
                h1_imag = np.concatenate([h1_imag_52k, h1_imag_30k, h1_imag_12k, h1_imag_6k, h1_imag_60k, h1_imag_40k, h1_imag_72k], axis=0)
                l1_imag = np.concatenate([l1_imag_52k, l1_imag_30k, l1_imag_12k, l1_imag_6k, l1_imag_60k, l1_imag_40k, l1_imag_72k], axis=0)
            
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
       
            # BBH dataset
            elif((self.dataset == 'BBH') and (snr_range_train == 'low')):
                f1 = h5py.File(data_config.BBH.path_train, 'r')
                f2 = h5py.File(data_config.BBH.path_train_low_SNR, 'r')
            
                h1_real_1 = abs(f1['h1_snr_series'][2000:200000][()])
                l1_real_1 = abs(f1['l1_snr_series'][2000:200000][()])
        
                h1_real_2 = abs(f2['h1_snr_series'][()])
                l1_real_2 = abs(f2['l1_snr_series'][()])
                
                h1_imag_1 = abs(np.imag(f1['h1_snr_series'][2000:200000][()]))
                l1_imag_1 = abs(np.imag(f1['l1_snr_series'][2000:200000][()]))
        
                h1_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
                l1_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
            
                h1_real = np.concatenate([h1_real_1, h1_real_2], axis=0)
                l1_real = np.concatenate([l1_real_1, l1_real_2], axis=0)
            
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2], axis=0)
            
                f1.close()
                f2.close()  
                        
        elif(self.train_negative_latency == True):
            
            if((self.dataset == 'BNS') and (self.train_negative_latency_seconds == '5')):
                
                f1 = h5py.File(data_config.BNS.path_train_2_det_negative_latency_5_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_2_det_negative_latency_5_2, 'r')
                
                h1_real_1 = abs(f1['h1_snr_series'][()])
                l1_real_1 = abs(f1['l1_snr_series'][()])
        
                h1_real_2 = abs(f2['h1_snr_series'][()])
                l1_real_2 = abs(f2['l1_snr_series'][()])
                
                h1_imag_1 = abs(np.imag(f1['h1_snr_series'][()]))
                l1_imag_1 = abs(np.imag(f1['l1_snr_series'][()]))
        
                h1_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
                l1_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
            
                h1_real = np.concatenate([h1_real_1, h1_real_2], axis=0)
                l1_real = np.concatenate([l1_real_1, l1_real_2], axis=0)
            
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2], axis=0)
            
                f1.close()
                f2.close()     
               
            elif((self.dataset == 'BNS') and (self.train_negative_latency_seconds == '10')):
                f1 = h5py.File(data_config.BNS.path_train_2_det_negative_latency_10, 'r')
                
                h1_real = abs(f1['h1_snr_series'][()])
                l1_real = abs(f1['l1_snr_series'][()])
                
                h1_imag = abs(np.imag(f1['h1_snr_series'][()]))
                l1_imag = abs(np.imag(f1['l1_snr_series'][()]))
            
                f1.close()                 
        
        
        h1_real = h1_real[:,:,None]
        l1_real = l1_real[:,:,None]
        
        h1_imag = h1_imag[:,:,None]
        l1_imag = l1_imag[:,:,None]
        
        X_train_real = np.concatenate((h1_real, l1_real), axis=2)
        X_train_imag = np.concatenate((h1_imag, l1_imag), axis=2)
        
        return X_train_real, X_train_imag
       
    
#    @staticmethod
    def load_train_3_det_parameters(self, data_config, snr_range_train, train_real):
        """Loads train parameters from path"""
        #NSBH
        if((self.dataset == 'NSBH') and (snr_range_train == 'high')):
            f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
            f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
            f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
            f4 = h5py.File(data_config.NSBH.path_train_4, 'r')
        
            ra_52k = 2.0*np.pi*f1['ra'][()]
            dec_52k = np.arcsin(1.0-2.0*f1['dec'][()])
            
#            ra_52k = f1['ra'][()]
#            dec_52k = f1['dec'][()]
            
            ra_30k = 2.0*np.pi*(f2['ra'][0:30000][()])
            dec_30k = np.arcsin(1.0-2.0*f2['dec'][0:30000][()])

#            ra_30k = f2['ra'][0:30000][()]
#            dec_30k = f2['dec'][0:30000][()]
        
            ra_12k = 2.0*np.pi*(f3['ra'][()])
            dec_12k = np.arcsin(1.0-2.0*f3['dec'][()])
            
#            ra_12k = f3['dec'][()]
#            dec_12k = f3['dec'][()]
            
            ra_6k = 2.0*np.pi*(f4['ra'][()])
            dec_6k = np.arcsin(1.0-2.0*f4['dec'][()])
#            ra_6k = f4['ra'][()] 
#            dec_6k = f4['dec'][()]
        
            ra = np.concatenate([ra_52k, ra_30k, ra_12k, ra_6k])
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            dec = np.concatenate([dec_52k, dec_30k, dec_12k, dec_6k])
            
        
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            
        if((self.dataset == 'NSBH') and (snr_range_train == 'low')):
            f1 = h5py.File(data_config.NSBH.path_train_low_snr_1, 'r')
            f2 = h5py.File(data_config.NSBH.path_train_low_snr_2, 'r')
        
            ra_60k = 2.0*np.pi*f1['ra'][()]
            dec_60k = np.arcsin(1.0-2.0*f1['dec'][()])
            
            ra_40k = 2.0*np.pi*(f2['ra'][0:40000][()])
            dec_40k = np.arcsin(1.0-2.0*f2['dec'][0:40000][()])
        
            ra = np.concatenate([ra_60k, ra_40k])
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            dec = np.concatenate([dec_60k, dec_40k])
            
            f1.close()
            f2.close()
        
        #BBH
        elif((self.dataset == 'BBH') and (snr_range_train == 'high')):
            f1 = h5py.File(data_config.BBH.path_train, 'r')

            ra = 2.0*np.pi*f1['ra'][0:100000][()]
            ra = ra - np.pi
            dec = np.arcsin(1.0-2.0*f1['dec'][0:100000][()])
#            ra = f1['ra'][0:100000][()]
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
#            dec = f1['dec'][0:100000][()]
        
            f1.close()
        
        elif((self.dataset == 'BBH') and (snr_range_train == 'low')):
            f1 = h5py.File(data_config.BBH.path_train_low_SNR, 'r')

            ra = 2.0*np.pi*f1['ra'][0:100000][()]
            ra = ra - np.pi
            dec = np.arcsin(1.0-2.0*f1['dec'][0:100000][()])
#            ra = f1['ra'][0:100000][()]
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
#            dec = f1['dec'][0:100000][()]
        
            f1.close()
        
        #BNS
        elif(self.dataset == 'BNS'):
            if((snr_range_train == 'high') and (train_real == False) and (self.train_negative_latency == False)):
                f1 = h5py.File(data_config.BNS.path_train_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_2, 'r')
            
                ra_22k = 2.0*np.pi*f1['ra'][0:22000][()]
                dec_22k = np.arcsin(1.0-2.0*f1['dec'][0:22000][()])

    #            ra_22k = f1['ra'][0:22000][()]
    #            dec_22k = f1['dec'][0:22000][()]
        
                ra_86k = 2.0*np.pi*f2['ra'][0:86000][()]
                dec_86k = np.arcsin(1.0-2.0*f2['dec'][0:86000][()])

    #            ra_86k = f2['ra'][0:86000][()]
    #            dec_86k = f2['dec'][0:86000][()]
            
                ra = np.concatenate([ra_22k, ra_86k], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_22k, dec_86k], axis=0)
        
                f1.close()
                f2.close()
            
            elif((snr_range_train == 'high') and (train_real == True) and (self.train_negative_latency == False)):
                
#                f1 = h5py.File(data_config.BNS.path_train_real_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_real_2, 'r')
                f3 = h5py.File(data_config.BNS.path_train_real_3, 'r')
            
#                ra_12k = 2.0*np.pi*f1['ra'][0:12000][()]
#                dec_12k = np.arcsin(1.0-2.0*f1['dec'][0:12000][()])

                ra_42k = 2.0*np.pi*f2['ra'][0:46000][()]
                dec_42k = np.arcsin(1.0-2.0*f2['dec'][0:46000][()])
        
                ra_46k = 2.0*np.pi*f3['ra'][0:42000][()]
                dec_46k = np.arcsin(1.0-2.0*f3['dec'][0:42000][()])

    #            ra_86k = f2['ra'][0:86000][()]
    #            dec_86k = f2['dec'][0:86000][()]
            
                ra = np.concatenate([ra_42k, ra_46k], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_42k, dec_46k], axis=0)
        
#                f1.close()
                f2.close()
                f3.close()
                
            elif((snr_range_train == 'low') and (train_real == False) and (self.train_negative_latency == False)):
                f1 = h5py.File(data_config.BNS.path_train_low_snr_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_low_snr_2, 'r')
                f3 = h5py.File(data_config.BNS.path_train_low_snr_3, 'r')
            
                ra_12k = 2.0*np.pi*f1['ra'][0:12000][()]
                dec_12k = np.arcsin(1.0-2.0*f1['dec'][0:12000][()])
        
                ra_36k = 2.0*np.pi*f2['ra'][0:36000][()]
                dec_36k = np.arcsin(1.0-2.0*f2['dec'][0:36000][()])
                
                ra_52k = 2.0*np.pi*f3['ra'][0:52000][()]
                dec_52k = np.arcsin(1.0-2.0*f3['dec'][0:52000][()])
            
                ra = np.concatenate([ra_12k, ra_36k, ra_52k], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_12k, dec_36k, dec_52k], axis=0)
        
                f1.close()
                f2.close()
                f3.close()
                
            elif((self.train_negative_latency == True) and (self.train_negative_latency_seconds == '5')):
                 
                f1 = h5py.File(data_config.BNS.path_train_5_sec)
              
                ra = 2.0*np.pi*f1['ra'][()]
                dec = np.arcsin(1.0-2.0*f1['dec'][()])
            
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
        
                f1.close()
                 
            elif((self.train_negative_latency == True) and (self.train_negative_latency_seconds == '10')):
                 
                f1 = h5py.File(data_config.BNS.path_train_10_sec)
              
                ra = 2.0*np.pi*f1['ra'][()]
                dec = np.arcsin(1.0-2.0*f1['dec'][()])
            
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
        
                f1.close()
                 
        ra = ra[:,None]
        ra_x = ra_x[:,None]
        ra_y = ra_y[:,None]
        
        dec = dec[:,None]

        y_train = np.concatenate((ra_x, ra_y, dec), axis=1)

        return y_train
   
    def load_train_2_det_parameters(self, data_config, snr_range_train):
        """Loads train parameters from path"""
        if(self.train_negative_latency == False):
            if((self.dataset == 'BNS') and (snr_range_train == 'low')):
            
                f1 = h5py.File(data_config.BNS.path_train_2_det_low_SNR, 'r')
                f2 = h5py.File(data_config.BNS.path_train_low_snr_1, 'r')
                f3 = h5py.File(data_config.BNS.path_train_low_snr_2, 'r')
                f4 = h5py.File(data_config.BNS.path_train_low_snr_3, 'r')
                f5 = h5py.File(data_config.BNS.path_train_1, 'r')
                f6 = h5py.File(data_config.BNS.path_train_2, 'r')
                f7 = h5py.File(data_config.BNS.path_train_2_det_low_SNR_1, 'r')
                f8 = h5py.File(data_config.BNS.path_train_2_det_high_SNR_1, 'r')
                f9 = h5py.File(data_config.BNS.path_train_2_det_high_SNR_2, 'r')
            
                ra = 2.0*np.pi*f1['ra'][()]
                dec = np.arcsin(1.0-2.0*f1['dec'][()])
            
                ra_12k = 2.0*np.pi*f2['ra'][0:12000][()]
                dec_12k = np.arcsin(1.0-2.0*f2['dec'][0:12000][()])
        
                ra_36k = 2.0*np.pi*f3['ra'][0:36000][()]
                dec_36k = np.arcsin(1.0-2.0*f3['dec'][0:36000][()])
                
                ra_52k = 2.0*np.pi*f4['ra'][0:52000][()]
                dec_52k = np.arcsin(1.0-2.0*f4['dec'][0:52000][()])
            
                ra_22k = 2.0*np.pi*f5['ra'][()]
                dec_22k = np.arcsin(1.0-2.0*f5['dec'][()])
            
                ra_86k = 2.0*np.pi*f6['ra'][()]
                dec_86k = np.arcsin(1.0-2.0*f6['dec'][()])
            
                ra_102k = 2.0*np.pi*f7['ra'][()]
                dec_102k = np.arcsin(1.0-2.0*f7['dec'][()])
            
                ra_high_1 = 2.0*np.pi*f8['ra'][()]
                dec_high_1 = np.arcsin(1.0-2.0*f8['dec'][()])
            
                ra_high_2 = 2.0*np.pi*f9['ra'][()]
                dec_high_2 = np.arcsin(1.0-2.0*f9['dec'][()])
                        
            
                ra = np.concatenate([ra, ra_12k, ra_36k, ra_52k, ra_22k, ra_86k, ra_102k], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec, dec_12k, dec_36k, dec_52k, dec_22k, dec_86k, dec_102k], axis=0)
        
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
                f8.close()
                f9.close()
            
            elif((self.dataset == 'NSBH') and (snr_range_train == 'low')):
            
                f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
                f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
                f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
                f4 = h5py.File(data_config.NSBH.path_train_4, 'r')
                f5 = h5py.File(data_config.NSBH.path_train_low_snr_1, 'r')
                f6 = h5py.File(data_config.NSBH.path_train_low_snr_2, 'r')
                f7 = h5py.File(data_config.NSBH.path_train_low_snr_3, 'r')
        
                ra_52k = 2.0*np.pi*f1['ra'][()]
                dec_52k = np.arcsin(1.0-2.0*f1['dec'][()])
            
                ra_30k = 2.0*np.pi*(f2['ra'][0:30000][()])
                dec_30k = np.arcsin(1.0-2.0*f2['dec'][0:30000][()])
        
                ra_12k = 2.0*np.pi*(f3['ra'][()])
                dec_12k = np.arcsin(1.0-2.0*f3['dec'][()])
            
                ra_6k = 2.0*np.pi*(f4['ra'][()])
                dec_6k = np.arcsin(1.0-2.0*f4['dec'][()])
            
                ra_60k = 2.0*np.pi*f5['ra'][()]
                dec_60k = np.arcsin(1.0-2.0*f5['dec'][()])
            
                ra_40k = 2.0*np.pi*(f6['ra'][0:40000][()])
                dec_40k = np.arcsin(1.0-2.0*f6['dec'][0:40000][()])
            
                ra_72k = 2.0*np.pi*(f7['ra'][()])
                dec_72k = np.arcsin(1.0-2.0*f7['dec'][()])
            
            
                ra = np.concatenate([ra_52k, ra_30k, ra_12k, ra_6k, ra_60k, ra_40k, ra_72k])
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_52k, dec_30k, dec_12k, dec_6k, dec_60k, dec_40k, dec_72k])
            
            
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
            
            elif((self.dataset == 'BBH') and (snr_range_train == 'low')):
            
                f1 = h5py.File(data_config.BBH.path_train, 'r')
                f2 = h5py.File(data_config.BBH.path_train_low_SNR, 'r')

                ra_1 = 2.0*np.pi*(f1['ra'][2000:200000][()])
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][2000:200000][()])
        
                ra_2 = 2.0*np.pi*(f2['ra'][()])
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
            
            
                ra = np.concatenate([ra_1, ra_2])
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_1, dec_2])            
            
                f1.close()
                f2.close()
                              
        elif((self.train_negative_latency == True)):
            
            if((self.dataset == 'BNS') and (self.train_negative_latency_seconds == '5')):
                
                f1 = h5py.File(data_config.BNS.path_train_2_det_negative_latency_5_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_2_det_negative_latency_5_2, 'r')
            
                ra_1 = 2.0*np.pi*(f1['ra'][()])
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
        
                ra_2 = 2.0*np.pi*(f2['ra'][()])
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                
            
                ra = np.concatenate([ra_1, ra_2])
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_1, dec_2])            
                
                f1.close()
                f2.close()
            
            elif((self.dataset == 'BNS') and (self.train_negative_latency_seconds == '10')):
                
                f1 = h5py.File(data_config.BNS.path_train_2_det_negative_latency_10, 'r')
            
                ra = 2.0*np.pi*(f1['ra'][()])
                dec = np.arcsin(1.0-2.0*f1['dec'][()])
        
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                        
            
                f1.close()

                                  
        ra = ra[:,None]
        ra_x = ra_x[:,None]
        ra_y = ra_y[:,None]
        
        dec = dec[:,None]

        y_train = np.concatenate((ra_x, ra_y, dec), axis=1)

        return y_train        
            
#    @staticmethod
    def load_test_3_det_data(self, data_config, test_real, snr_range_test, test_negative_latency):
        """Loads dataset from path"""
        #Get the HDF5 group
        if(test_negative_latency == False):
        #NSBH
            if(self.dataset == 'NSBH'):
                if(snr_range_test == 'high'):
                    f_test = h5py.File(data_config.NSBH.path_test, 'r')
                    
                if((test_real == True) and (snr_range_test == 'low')):
                    
                    f1 = h5py.File(data_config.NSBH.path_test_GW190917, 'r')
                    
                    h1_test_real = abs(f1['h1_snr_series'][()])
                    l1_test_real = abs(f1['l1_snr_series'][()])
                    v1_test_real = abs(f1['v1_snr_series'][()])
                    
                    h1_test_imag = abs(np.imag(f1['h1_snr_series'][()]))
                    l1_test_imag = abs(np.imag(f1['l1_snr_series'][()]))
                    v1_test_imag = abs(np.imag(f1['v1_snr_series'][()]))
                    
                    f1.close()
                
                if((test_real == False) and snr_range_test == 'low'):
                    
                    f_test = h5py.File(data_config.NSBH.path_test_low_snr, 'r')
                   
                    group_test = f_test['omf_injection_snr_samples']
        
                    data_h1_test = group_test['h1_snr']
                    data_l1_test = group_test['l1_snr']
                    data_v1_test = group_test['v1_snr']
        
                    h1_test_real = np.zeros([self.n_test, self.n_samples])
                    l1_test_real = np.zeros([self.n_test, self.n_samples])
                    v1_test_real = np.zeros([self.n_test, self.n_samples])
        
                    h1_test_imag = np.zeros([self.n_test, self.n_samples])
                    l1_test_imag = np.zeros([self.n_test, self.n_samples])
                    v1_test_imag = np.zeros([self.n_test, self.n_samples])
        
                    for i in range(self.n_test):
                        h1_test_real[i] = abs(data_h1_test[str(i)][()][1840:2250] )
                        l1_test_real[i] = abs(data_l1_test[str(i)][()][1840:2250] )
                        v1_test_real[i] = abs(data_v1_test[str(i)][()][1840:2250] )
    
                        h1_test_imag[i] = abs(np.imag(data_h1_test[str(i)][()][1840:2250]))
                        l1_test_imag[i] = abs(np.imag(data_l1_test[str(i)][()][1840:2250]))
                        v1_test_imag[i] = abs(np.imag(data_v1_test[str(i)][()][1840:2250]))
        
                    f_test.close()                    
            
            elif(self.dataset == 'BBH'):
                if(test_real == True):
                    f1 = h5py.File(data_config.BBH.path_test_GW170729, 'r')
                    f2 = h5py.File(data_config.BBH.path_test_GW170809, 'r')
                    f3 = h5py.File(data_config.BBH.path_test_GW170814, 'r')
                    f4 = h5py.File(data_config.BBH.path_test_GW170818, 'r')
                    
                    h1_real_1 = abs(f1['h1_snr_series'][()])
                    l1_real_1 = abs(f1['l1_snr_series'][()])
                    v1_real_1 = abs(f1['v1_snr_series'][()])
        
                    h1_real_2 = abs(f2['h1_snr_series'][()])
                    l1_real_2 = abs(f2['l1_snr_series'][()])
                    v1_real_2 = abs(f2['v1_snr_series'][()])
                    
                    h1_real_3 = abs(f3['h1_snr_series'][()])
                    l1_real_3 = abs(f3['l1_snr_series'][()])
                    v1_real_3 = abs(f3['v1_snr_series'][()])
        
                    h1_real_4 = abs(f4['h1_snr_series'][()])
                    l1_real_4 = abs(f4['l1_snr_series'][()])
                    v1_real_4 = abs(f4['v1_snr_series'][()])
                    
                    
                    h1_imag_1 = abs(np.imag(f1['h1_snr_series'][()]))
                    l1_imag_1 = abs(np.imag(f1['l1_snr_series'][()]))
                    v1_imag_1 = abs(np.imag(f1['v1_snr_series'][()]))
        
                    h1_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
                    l1_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
                    v1_imag_2 = abs(np.imag(f2['v1_snr_series'][()]))
                    
                    h1_imag_3 = abs(np.imag(f3['h1_snr_series'][()]))
                    l1_imag_3 = abs(np.imag(f3['l1_snr_series'][()]))
                    v1_imag_3 = abs(np.imag(f3['v1_snr_series'][()]))
        
                    h1_imag_4 = abs(np.imag(f4['h1_snr_series'][()]))
                    l1_imag_4 = abs(np.imag(f4['l1_snr_series'][()]))
                    v1_imag_4 = abs(np.imag(f4['v1_snr_series'][()]))
                    
                    h1_real_1 = h1_real_1[None,:]
                    h1_real_2 = h1_real_2[None,:]
                    h1_real_3 = h1_real_3[None,:]
                    h1_real_4 = h1_real_4[None,:]

                    l1_real_1 = l1_real_1[None,:]
                    l1_real_2 = l1_real_2[None,:]
                    l1_real_3 = l1_real_3[None,:]
                    l1_real_4 = l1_real_4[None,:]

                    v1_real_1 = v1_real_1[None,:]
                    v1_real_2 = v1_real_2[None,:]
                    v1_real_3 = v1_real_3[None,:]
                    v1_real_4 = v1_real_4[None,:]

                    h1_imag_1 = h1_imag_1[None,:]
                    h1_imag_2 = h1_imag_2[None,:]
                    h1_imag_3 = h1_imag_3[None,:]
                    h1_imag_4 = h1_imag_4[None,:]

                    l1_imag_1 = l1_imag_1[None,:]
                    l1_imag_2 = l1_imag_2[None,:]
                    l1_imag_3 = l1_imag_3[None,:]
                    l1_imag_4 = l1_imag_4[None,:]

                    v1_imag_1 = v1_imag_1[None,:]
                    v1_imag_2 = v1_imag_2[None,:]
                    v1_imag_3 = v1_imag_3[None,:]
                    v1_imag_4 = v1_imag_4[None,:]
            
                    h1_test_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4], axis=0)
                    l1_test_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4], axis=0)
                    v1_test_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4], axis=0)
            
                    h1_test_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4], axis=0)
                    l1_test_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4], axis=0)
                    v1_test_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4], axis=0)
                    
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    
                elif((test_real == False) and (snr_range_test == 'high')):
                    f_test = h5py.File(data_config.BBH.path_test, 'r')
                    
                elif((test_real == False) and (snr_range_test == 'low')):
                    
                    f1 = h5py.File(data_config.BBH.path_test_low_SNR, 'r')

                    h1_test_real = abs(f1['h1_snr_series'][0:2000][()])
                    l1_test_real = abs(f1['l1_snr_series'][0:2000][()])
                    v1_test_real = abs(f1['v1_snr_series'][0:2000][()])
       
                    h1_test_imag = abs(np.imag(f1['h1_snr_series'][0:2000][()]))
                    l1_test_imag = abs(np.imag(f1['l1_snr_series'][0:2000][()]))
                    v1_test_imag = abs(np.imag(f1['v1_snr_series'][0:2000][()]))
            
                    f1.close()
                    
            elif(self.dataset == 'BNS'):
                if(test_real == True):
                    
                    f_test = h5py.File(data_config.BNS.path_test_GW170817, 'r')
                    
                    h1_test_real = abs(f_test['h1_snr_series'][()])
                    l1_test_real = abs(f_test['l1_snr_series'][()])
                    v1_test_real = abs(f_test['v1_snr_series'][()])
        
                    h1_test_imag = abs(np.imag(f_test['h1_snr_series'][()]))
                    l1_test_imag = abs(np.imag(f_test['l1_snr_series'][()]))
                    v1_test_imag = abs(np.imag(f_test['v1_snr_series'][()]))
                    
                    h1_test_real = h1_test_real[None,:]
                    l1_test_real = l1_test_real[None,:]
                    v1_test_real = v1_test_real[None,:]
                    
                    h1_test_imag = h1_test_imag[None,:]
                    l1_test_imag = l1_test_imag[None,:]
                    v1_test_imag = v1_test_imag[None,:]
                    
                    f_test.close()
                
                    
#                    group_test = f_test['omf_injection_snr_samples']
        
#                    data_h1_test = group_test['h1_snr']
#                    data_l1_test = group_test['l1_snr']
#                    data_v1_test = group_test['v1_snr']
        
#                    h1_test_real = np.zeros([self.n_test, self.n_samples])
#                    l1_test_real = np.zeros([self.n_test, self.n_samples])
#                    v1_test_real = np.zeros([self.n_test, self.n_samples])
        
#                    h1_test_imag = np.zeros([self.n_test, self.n_samples])
#                    l1_test_imag = np.zeros([self.n_test, self.n_samples])
#                    v1_test_imag = np.zeros([self.n_test, self.n_samples])
        
#                    for i in range(self.n_test):
#                        h1_test_real[i] = abs(data_h1_test[str(i)][()][1840:2250] )
#                        l1_test_real[i] = abs(data_l1_test[str(i)][()][1840:2250] )
#                        v1_test_real[i] = abs(data_v1_test[str(i)][()][1840:2250] )
    
#                        h1_test_imag[i] = abs(np.imag(data_h1_test[str(i)][()][1840:2250]))
#                        l1_test_imag[i] = abs(np.imag(data_l1_test[str(i)][()][1840:2250]))
#                        v1_test_imag[i] = abs(np.imag(data_v1_test[str(i)][()][1840:2250]))
        
#                    f_test.close()
                            
                elif((test_real == False) and (snr_range_test == 'high')):
                    f_test = h5py.File(data_config.BNS.path_test, 'r')
                
                elif((test_real == False) and (snr_range_test == 'low')):
                    
                    f_test = h5py.File(data_config.BNS.path_test_low_SNR, 'r')
                    
                    group_test = f_test['omf_injection_snr_samples']
        
                    data_h1_test = group_test['h1_snr']
                    data_l1_test = group_test['l1_snr']
                    data_v1_test = group_test['v1_snr']
        
                    h1_test_real = np.zeros([self.n_test, self.n_samples])
                    l1_test_real = np.zeros([self.n_test, self.n_samples])
                    v1_test_real = np.zeros([self.n_test, self.n_samples])
        
                    h1_test_imag = np.zeros([self.n_test, self.n_samples])
                    l1_test_imag = np.zeros([self.n_test, self.n_samples])
                    v1_test_imag = np.zeros([self.n_test, self.n_samples])
        
                    for i in range(self.n_test):
                        h1_test_real[i] = abs(data_h1_test[str(i)][()][1840:2250] )
                        l1_test_real[i] = abs(data_l1_test[str(i)][()][1840:2250] )
                        v1_test_real[i] = abs(data_v1_test[str(i)][()][1840:2250] )
    
                        h1_test_imag[i] = abs(np.imag(data_h1_test[str(i)][()][1840:2250]))
                        l1_test_imag[i] = abs(np.imag(data_l1_test[str(i)][()][1840:2250]))
                        v1_test_imag[i] = abs(np.imag(data_v1_test[str(i)][()][1840:2250]))
        
                    f_test.close()
        
        elif(test_negative_latency == True):
            
            if(self.dataset == 'BNS'):
                f1 = h5py.File(data_config.BNS.path_test_3_det_0_secs, 'r')
                f2 = h5py.File(data_config.BNS.path_test_3_det_5_secs, 'r')
                f3 = h5py.File(data_config.BNS.path_test_3_det_10_secs, 'r')
                f4 = h5py.File(data_config.BNS.path_test_3_det_15_secs, 'r')
            
                h1_test_real_1 = abs(f1['h1_snr_series'][()])
                l1_test_real_1 = abs(f1['l1_snr_series'][()])
                v1_test_real_1 = abs(f1['v1_snr_series'][()])
        
                h1_test_imag_1 = abs(np.imag(f1['h1_snr_series'][()]))
                l1_test_imag_1 = abs(np.imag(f1['l1_snr_series'][()]))
                v1_test_imag_1 = abs(np.imag(f1['v1_snr_series'][()]))
                        
                h1_test_real_2 = abs(f2['h1_snr_series'][()])
                l1_test_real_2 = abs(f2['l1_snr_series'][()])
                v1_test_real_2 = abs(f2['v1_snr_series'][()])
        
                h1_test_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
                l1_test_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
                v1_test_imag_2 = abs(np.imag(f2['v1_snr_series'][()]))
            
                h1_test_real_3 = abs(f3['h1_snr_series'][()])
                l1_test_real_3 = abs(f3['l1_snr_series'][()])
                v1_test_real_3 = abs(f3['v1_snr_series'][()])
        
                h1_test_imag_3 = abs(np.imag(f3['h1_snr_series'][()]))
                l1_test_imag_3 = abs(np.imag(f3['l1_snr_series'][()]))
                v1_test_imag_3 = abs(np.imag(f3['v1_snr_series'][()]))
                        
                h1_test_real_4 = abs(f4['h1_snr_series'][()])
                l1_test_real_4 = abs(f4['l1_snr_series'][()])
                v1_test_real_4 = abs(f4['v1_snr_series'][()])
        
                h1_test_imag_4 = abs(np.imag(f4['h1_snr_series'][()]))
                l1_test_imag_4 = abs(np.imag(f4['l1_snr_series'][()]))
                v1_test_imag_4 = abs(np.imag(f4['v1_snr_series'][()]))
            
                h1_test_real = np.concatenate([h1_test_real_1, h1_test_real_2, h1_test_real_3, h1_test_real_4], axis=0)
                l1_test_real = np.concatenate([l1_test_real_1, l1_test_real_2, l1_test_real_3, l1_test_real_4], axis=0)
                v1_test_real = np.concatenate([v1_test_real_1, v1_test_real_2, v1_test_real_3, v1_test_real_4], axis=0)
            
                h1_test_imag = np.concatenate([h1_test_imag_1, h1_test_imag_2, h1_test_imag_3, h1_test_imag_4], axis=0)
                l1_test_imag = np.concatenate([l1_test_imag_1, l1_test_imag_2, l1_test_imag_3, l1_test_imag_4], axis=0)
                v1_test_imag = np.concatenate([v1_test_imag_1, v1_test_imag_2, v1_test_imag_3, v1_test_imag_4], axis=0)
            
                f1.close()
                f2.close()
                f3.close()
                f4.close()
        
        h1_test_real = h1_test_real[:,:,None]
        l1_test_real = l1_test_real[:,:,None]
        v1_test_real = v1_test_real[:,:,None]
    
        h1_test_imag = h1_test_imag[:,:,None]
        l1_test_imag = l1_test_imag[:,:,None]
        v1_test_imag = v1_test_imag[:,:,None]
        
        X_test_real = np.concatenate((h1_test_real, l1_test_real, v1_test_real), axis=2)
        X_test_imag = np.concatenate((h1_test_imag, l1_test_imag, v1_test_imag), axis=2)
    
        return X_test_real, X_test_imag
    
    def load_test_2_det_data(self, data_config, test_real, snr_range_test, test_negative_latency):
        """Loads dataset from path"""
        #Get the HDF5 group
        #BNS
        if((self.dataset == 'BNS') and (snr_range_test == 'low') and (test_negative_latency == False)):
            f1 = h5py.File(data_config.BNS.path_test_2_det_low_SNR, 'r')
            f2 = h5py.File(data_config.BNS.path_test, 'r')
            f3 = h5py.File(data_config.BNS.path_test_2_det_high_SNR, 'r')

            h1_test_real_1 = abs(f1['h1_snr_series'][()])
            l1_test_real_1 = abs(f1['l1_snr_series'][()])
        
            h1_test_imag_1 = abs(np.imag(f1['h1_snr_series'][()]))
            l1_test_imag_1 = abs(np.imag(f1['l1_snr_series'][()]))
                        
            h1_test_real_2 = abs(f2['h1_snr_series'][()])
            l1_test_real_2 = abs(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
            l1_test_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
            
            h1_test_real_3 = abs(f3['h1_snr_series'][()])
            l1_test_real_3 = abs(f3['l1_snr_series'][()])
        
            h1_test_imag_3 = abs(np.imag(f3['h1_snr_series'][()]))
            l1_test_imag_3 = abs(np.imag(f3['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_1, h1_test_real_2, h1_test_real_3], axis=0)
            l1_test_real = np.concatenate([l1_test_real_1, l1_test_real_2, l1_test_real_3], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_1, h1_test_imag_2, h1_test_imag_3], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_1, l1_test_imag_2, l1_test_imag_3], axis=0)
            
            f1.close()
            f2.close()
            f3.close()
            
        if((self.dataset == 'BNS') and (test_negative_latency == True)):
#            f1 = h5py.File(data_config.BNS.path_test_2_det_0_secs, 'r')
            f2 = h5py.File(data_config.BNS.path_test_2_det_5_secs, 'r')
            f3 = h5py.File(data_config.BNS.path_test_2_det_10_secs, 'r')
            f4 = h5py.File(data_config.BNS.path_test_2_det_15_secs, 'r')
            
#            h1_test_real_1 = abs(f1['h1_snr_series'][()])
#            l1_test_real_1 = abs(f1['l1_snr_series'][()])
        
#            h1_test_imag_1 = abs(np.imag(f1['h1_snr_series'][()]))
#            l1_test_imag_1 = abs(np.imag(f1['l1_snr_series'][()]))
                        
            h1_test_real_2 = abs(f2['h1_snr_series'][()])
            l1_test_real_2 = abs(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
            l1_test_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
            
            h1_test_real_3 = abs(f3['h1_snr_series'][()])
            l1_test_real_3 = abs(f3['l1_snr_series'][()])
        
            h1_test_imag_3 = abs(np.imag(f3['h1_snr_series'][()]))
            l1_test_imag_3 = abs(np.imag(f3['l1_snr_series'][()]))
                        
            h1_test_real_4 = abs(f4['h1_snr_series'][()])
            l1_test_real_4 = abs(f4['l1_snr_series'][()])
        
            h1_test_imag_4 = abs(np.imag(f4['h1_snr_series'][()]))
            l1_test_imag_4 = abs(np.imag(f4['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_2, h1_test_real_3, h1_test_real_4], axis=0)
            l1_test_real = np.concatenate([l1_test_real_2, l1_test_real_3, l1_test_real_4], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_2, h1_test_imag_3, h1_test_imag_4], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_2, l1_test_imag_3, l1_test_imag_4], axis=0)
            
#            f1.close()
            f2.close()
            f3.close()
            f4.close()
            
            
        #NSBH
        elif((self.dataset == 'NSBH') and (snr_range_test == 'low')):
            f1 = h5py.File(data_config.NSBH.path_test, 'r')
            f2 = h5py.File(data_config.NSBH.path_test_low_snr, 'r')

            h1_test_real_1 = abs(f1['h1_snr_series'][()])
            l1_test_real_1 = abs(f1['l1_snr_series'][()])
        
            h1_test_imag_1 = abs(np.imag(f1['h1_snr_series'][()]))
            l1_test_imag_1 = abs(np.imag(f1['l1_snr_series'][()]))
                        
            h1_test_real_2 = abs(f2['h1_snr_series'][()])
            l1_test_real_2 = abs(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
            l1_test_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_1, h1_test_real_2], axis=0)
            l1_test_real = np.concatenate([l1_test_real_1, l1_test_real_2], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_1, h1_test_imag_2], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_1, l1_test_imag_2], axis=0)
            
            f1.close()
            f2.close()
            
        #BBH
        elif((self.dataset == 'BBH') and (snr_range_test == 'low')):
            f1 = h5py.File(data_config.BBH.path_train, 'r')
            f2 = h5py.File(data_config.BBH.path_test_low_SNR, 'r')

            h1_test_real_1 = abs(f1['h1_snr_series'][()])
            l1_test_real_1 = abs(f1['l1_snr_series'][()])
       
            h1_test_imag_1 = abs(np.imag(f1['h1_snr_series'][()]))
            l1_test_imag_1 = abs(np.imag(f1['l1_snr_series'][()]))
                        
            h1_test_real_2 = abs(f2['h1_snr_series'][()])
            l1_test_real_2 = abs(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = abs(np.imag(f2['h1_snr_series'][()]))
            l1_test_imag_2 = abs(np.imag(f2['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_1, h1_test_real_2], axis=0)
            l1_test_real = np.concatenate([l1_test_real_1, l1_test_real_2], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_1, h1_test_imag_2], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_1, l1_test_imag_2], axis=0)
            
            f1.close()
            f2.close()
        
        
        h1_test_real = h1_test_real[:,:,None]
        l1_test_real = l1_test_real[:,:,None]
    
        h1_test_imag = h1_test_imag[:,:,None]
        l1_test_imag = l1_test_imag[:,:,None]
        
        X_test_real = np.concatenate((h1_test_real, l1_test_real), axis=2)
        X_test_imag = np.concatenate((h1_test_imag, l1_test_imag), axis=2)
    
        return X_test_real, X_test_imag
                           
#    @staticmethod
    def load_test_3_det_parameters(self, data_config, test_real, snr_range_test, test_negative_latency):
        """Loads train parameters from path"""
        if(test_negative_latency == False):
            if(self.dataset == 'NSBH'):
                if(snr_range_test == 'high'):
                    f_test = h5py.File(data_config.NSBH.path_test, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                    
                elif((test_real == False) and (snr_range_test == 'low')):
                    f_test = h5py.File(data_config.NSBH.path_test_low_snr, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                
                elif((test_real == True) and (snr_range_test == 'low')):
                    
                    f_test = h5py.File(data_config.NSBH.path_test_GW190917, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                    
            
            elif(self.dataset == 'BBH'):
                if(test_real == True):
                    f1 = h5py.File(data_config.BBH.path_test_GW170729, 'r')
                    f2 = h5py.File(data_config.BBH.path_test_GW170809, 'r')
                    f3 = h5py.File(data_config.BBH.path_test_GW170814, 'r')
                    f4 = h5py.File(data_config.BBH.path_test_GW170818, 'r')
                    
                    ra_1 = 2.0*np.pi*(f1['ra'][()])
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
        
                    ra_2 = 2.0*np.pi*(f2['ra'][()])
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                
                    ra_3 = 2.0*np.pi*(f3['ra'][()])
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
        
                    ra_4 = 2.0*np.pi*(f4['ra'][()])
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                
                    ra_test = np.concatenate([ra_1, ra_2, ra_3, ra_4])
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
            
                    dec_test = np.concatenate([dec_1, dec_2, dec_3, dec_4])            
                
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    
                elif((test_real == False) and (snr_range_test == 'high')):
                    f_test = h5py.File(data_config.BBH.path_test, 'r')
                elif((test_real == False) and (snr_range_test == 'low')):
                    f_test = h5py.File(data_config.BBH.path_test_low_SNR, 'r')
            
            elif(self.dataset == 'BNS'):
            
                if(test_real == True):
                
                    f_test = h5py.File(data_config.BNS.path_test_GW170817, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                
                elif((test_real == False) and (snr_range_test == 'high')):
                
                    f_test = h5py.File(data_config.BNS.path_test, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                
                elif((test_real == False) and (snr_range_test == 'low')):
                
                    f_test = h5py.File(data_config.BNS.path_test_low_SNR, 'r')
                            
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
        
        elif(test_negative_latency == True):
            
            if(self.dataset == 'BNS'):
            
                f_test_1 = h5py.File(data_config.BNS.path_test_3_det_0_secs, 'r')
                f_test_2 = h5py.File(data_config.BNS.path_test_3_det_0_secs, 'r')
                f_test_3 = h5py.File(data_config.BNS.path_test_3_det_0_secs, 'r')
                f_test_4 = h5py.File(data_config.BNS.path_test_3_det_0_secs, 'r')
            
                ra_1 = 2.0*np.pi*f_test_1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
            
                ra_2 = 2.0*np.pi*f_test_2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
            
                ra_3 = 2.0*np.pi*f_test_3['ra'][()]
                dec_3 = np.arcsin(1.0-2.0*f_test_3['dec'][()])
            
                ra_4 = 2.0*np.pi*f_test_4['ra'][()]
                dec_4 = np.arcsin(1.0-2.0*f_test_4['dec'][()])
                        
                ra_test = np.concatenate([ra_1, ra_2, ra_3, ra_4], axis=0)
                ra_test = ra_test - np.pi
                ra_test_x = np.cos(ra_test)
                ra_test_y = np.sin(ra_test)
            
                dec_test = np.concatenate([dec_1, dec_2, dec_3, dec_4], axis=0)

                f_test_1.close()
                f_test_2.close()
                f_test_3.close()
                f_test_4.close()
            
#        ra_test = data_ra
#        dec_test = data_dec
        
        ra_test = ra_test[:,None]
        ra_test_x = ra_test_x[:, None]
        ra_test_y = ra_test_y[:, None]
        
        dec_test = dec_test[:,None]

        y_test = np.concatenate((ra_test_x, ra_test_y, dec_test), axis=1)

        return y_test, ra_test_x, ra_test_y, ra_test, dec_test
    
    def load_test_2_det_parameters(self, data_config, test_real, snr_range_test, test_negative_latency):
        """Loads train parameters from path"""
        if((self.dataset == 'BNS') and (snr_range_test == 'low')):
            
            f_test_1 = h5py.File(data_config.BNS.path_test_2_det_low_SNR, 'r')
            f_test_2 = h5py.File(data_config.BNS.path_test, 'r')
            f_test_3 = h5py.File(data_config.BNS.path_test_2_det_high_SNR, 'r')
            
            ra_1 = 2.0*np.pi*f_test_1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
            
            ra_2 = 2.0*np.pi*f_test_2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
            
            ra_3 = 2.0*np.pi*f_test_3['ra'][()]
            dec_3 = np.arcsin(1.0-2.0*f_test_3['dec'][()])
                        
            ra_test = np.concatenate([ra_1, ra_2, ra_3], axis=0)
            ra_test = ra_test - np.pi
            ra_test_x = np.cos(ra_test)
            ra_test_y = np.sin(ra_test)
            
            dec_test = np.concatenate([dec_1, dec_2, dec_3], axis=0)

            f_test_1.close()
            f_test_2.close()
            f_test_3.close()
            
        if((self.dataset == 'BNS') and (test_negative_latency == True)):
            
#            f_test_1 = h5py.File(data_config.BNS.path_test_2_det_0_secs, 'r')
            f_test_2 = h5py.File(data_config.BNS.path_test_2_det_0_secs, 'r')
            f_test_3 = h5py.File(data_config.BNS.path_test_2_det_0_secs, 'r')
            f_test_4 = h5py.File(data_config.BNS.path_test_2_det_0_secs, 'r')
            
#            ra_1 = 2.0*np.pi*f_test_1['ra'][()]
#            dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
            
            ra_2 = 2.0*np.pi*f_test_2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
            
            ra_3 = 2.0*np.pi*f_test_3['ra'][()]
            dec_3 = np.arcsin(1.0-2.0*f_test_3['dec'][()])
            
            ra_4 = 2.0*np.pi*f_test_4['ra'][()]
            dec_4 = np.arcsin(1.0-2.0*f_test_4['dec'][()])
                        
            ra_test = np.concatenate([ra_2, ra_3, ra_4], axis=0)
            ra_test = ra_test - np.pi
            ra_test_x = np.cos(ra_test)
            ra_test_y = np.sin(ra_test)
            
            dec_test = np.concatenate([dec_2, dec_3, dec_4], axis=0)

#            f_test_1.close()
            f_test_2.close()
            f_test_3.close()
            f_test_4.close()
            
        elif((self.dataset == 'NSBH') and (snr_range_test == 'low')):
            
            f_test_1 = h5py.File(data_config.NSBH.path_test, 'r')
            f_test_2 = h5py.File(data_config.NSBH.path_test_low_snr, 'r')
            
            ra_1 = 2.0*np.pi*f_test_1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
            
            ra_2 = 2.0*np.pi*f_test_2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
                        
            ra_test = np.concatenate([ra_1, ra_2], axis=0)
            ra_test = ra_test - np.pi
            ra_test_x = np.cos(ra_test)
            ra_test_y = np.sin(ra_test)
            
            dec_test = np.concatenate([dec_1, dec_2], axis=0)

            f_test_1.close()
            f_test_2.close()
            
        elif((self.dataset == 'BBH') and (snr_range_test == 'low')):
            
            f_test_1 = h5py.File(data_config.BBH.path_train, 'r')
            f_test_2 = h5py.File(data_config.BBH.path_test_low_SNR, 'r')
            
            ra_1 = 2.0*np.pi*f_test_1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
            
            ra_2 = 2.0*np.pi*f_test_2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
                        
            ra_test = np.concatenate([ra_1, ra_2], axis=0)
            ra_test = ra_test - np.pi
            ra_test_x = np.cos(ra_test)
            ra_test_y = np.sin(ra_test)
            
            dec_test = np.concatenate([dec_1, dec_2], axis=0)

            f_test_1.close()
            f_test_2.close()

        ra_test = ra_test[:,None]
        ra_test_x = ra_test_x[:, None]
        ra_test_y = ra_test_y[:, None]
        
        dec_test = dec_test[:,None]

        y_test = np.concatenate((ra_test_x, ra_test_y, dec_test), axis=1)

        return y_test, ra_test_x, ra_test_y, ra_test, dec_test
    
    
#    @staticmethod
    def load_3_det_samples(self, data_config, X_real, X_imag, y, num_samples, snr_range_train, snr_range_test, data):
        """Loads 3 det samples and parameters from path"""
        if(self.dataset == 'NSBH'):
            
            if((data == 'train') and (snr_range_train=='high')):
                f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
                f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
                f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
                f4 = h5py.File(data_config.NSBH.path_train_4, 'r')            
                
        
                h1_snr_1 = f1['H1_SNR'][()]
                h1_snr_2 = f2['H1_SNR'][0:30000][()]
                h1_snr_3 = f3['H1_SNR'][()]
                h1_snr_4 = f4['H1_SNR'][()]            
                

                h1_snr = np.concatenate([h1_snr_1, h1_snr_2, h1_snr_3, h1_snr_4])
        
                l1_snr_1 = f1['L1_SNR'][()]
                l1_snr_2 = f2['L1_SNR'][0:30000][()]
                l1_snr_3 = f3['L1_SNR'][()]
                l1_snr_4 = f4['L1_SNR'][()]            
                

                l1_snr = np.concatenate([l1_snr_1, l1_snr_2, l1_snr_3, l1_snr_4])

                v1_snr_1 = f1['V1_SNR'][()]
                v1_snr_2 = f2['V1_SNR'][0:30000][()]
                v1_snr_3 = f3['V1_SNR'][()]
                v1_snr_4 = f4['V1_SNR'][()]            
                

                v1_snr = np.concatenate([v1_snr_1, v1_snr_2, v1_snr_3, v1_snr_4])
        
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
            
                ra_1 = 2.0*np.pi*f1['ra'][()]
                ra_2 = 2.0*np.pi*f2['ra'][0:30000][()]
                ra_3 = 2.0*np.pi*f3['ra'][()]
                ra_4 = 2.0*np.pi*f4['ra'][()]
                
#                ra_1 = f1['ra'][()]
#                ra_2 = f2['ra'][0:30000][()]
#                ra_3 = f3['ra'][()]
#                ra_4 = f4['ra'][()]
                
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][0:30000][()])
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])

#                dec_1 = f1['dec'][()]
#                dec_2 = f2['dec'][0:30000][()]
#                dec_3 = f3['dec'][()]
#                dec_4 = f4['dec'][()]
            
                ra = np.concatenate([ra_1, ra_2, ra_3, ra_4])
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.concatenate([dec_1, dec_2, dec_3, dec_4])

    
                f1.close()
                f2.close()
                f3.close()
                f4.close()
            
            elif((data == 'train') and (snr_range_train=='low')):
                f1 = h5py.File(data_config.NSBH.path_train_low_snr_1, 'r')
                f2 = h5py.File(data_config.NSBH.path_train_low_snr_2, 'r')      
                        
                h1_snr_1 = f1['H1_SNR'][()]
                h1_snr_2 = f2['H1_SNR'][()]

                h1_snr = np.concatenate([h1_snr_1, h1_snr_2])
        
                l1_snr_1 = f1['L1_SNR'][()]
                l1_snr_2 = f2['L1_SNR'][()]       
                
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2])

                v1_snr_1 = f1['V1_SNR'][()]
                v1_snr_2 = f2['V1_SNR'][()]       
                
                v1_snr = np.concatenate([v1_snr_1, v1_snr_2])
        
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
            
                ra_1 = 2.0*np.pi*f1['ra'][()]
                ra_2 = 2.0*np.pi*f2['ra'][()]
                
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
            
                ra = np.concatenate([ra_1, ra_2])
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.concatenate([dec_1, dec_2])
    
                f1.close()
                f2.close()
            
            elif(data == 'test'):
                if(snr_range_test == 'high'):
                    f_test = h5py.File(data_config.NSBH.path_test, 'r')
                    h1_snr = f_test['H1_SNR'][()]
                    l1_snr = f_test['L1_SNR'][()]
                    v1_snr = f_test['V1_SNR'][()]
            
                    h1 = h1_snr > self.min_snr
                    l1 = l1_snr > self.min_snr
                    v1 = v1_snr > self.min_snr
            
                    ra = 2.0*np.pi*f_test['ra'][()]
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                
                    dec = np.arcsin(1.0-2.0*f_test['dec'][()])

    #                ra = f_test['ra'][()]
    #                dec = f_test['dec'][()]

                    f_test.close()
        
                elif(snr_range_test == 'low'):
                    f_test = h5py.File(data_config.NSBH.path_test_low_snr, 'r')
                    h1_snr = f_test['H1_SNR'][()]
                    l1_snr = f_test['L1_SNR'][()]
                    v1_snr = f_test['V1_SNR'][()]
            
                    h1 = h1_snr > self.min_snr
                    l1 = l1_snr > self.min_snr
                    v1 = v1_snr > self.min_snr
            
                    ra = 2.0*np.pi*f_test['ra'][()]
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                
                    dec = np.arcsin(1.0-2.0*f_test['dec'][()])

                    f_test.close()
                
        
        elif(self.dataset == 'BBH'):
            if((data == 'train') and (snr_range_train=='high')):
                f1 = h5py.File(data_config.BBH.path_train, 'r')
                
                h1_snr = f1['H1_SNR'][0:100000][()]
                l1_snr = f1['L1_SNR'][0:100000][()]
                v1_snr = f1['V1_SNR'][0:100000][()]
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
        
                ra = 2.0*np.pi*f1['ra'][0:100000][()]
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.arcsin(1.0-2.0*f1['dec'][0:100000][()])
                
#                ra = f1['ra'][()]
#                dec = f1['dec'][()]
                
                f1.close()
        
            elif((data == 'train') and (snr_range_train=='low')):
                f1 = h5py.File(data_config.BBH.path_train_low_SNR, 'r')
                
                h1_snr = f1['H1_SNR'][0:100000][()]
                l1_snr = f1['L1_SNR'][0:100000][()]
                v1_snr = f1['V1_SNR'][0:100000][()]
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
        
                ra = 2.0*np.pi*f1['ra'][0:100000][()]
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.arcsin(1.0-2.0*f1['dec'][0:100000][()])
                
#                ra = f1['ra'][()]
#                dec = f1['dec'][()]
                
                f1.close()
                
            elif(data == 'test'):
                if(snr_range_test == 'high'):
                    f1 = h5py.File(data_config.BBH.path_test, 'r')
                elif(snr_range_test == 'low'):
                    f1 = h5py.File(data_config.BBH.path_test_low_SNR, 'r')
                
                h1_snr = f1['H1_SNR'][()]
                l1_snr = f1['L1_SNR'][()]
                v1_snr = f1['V1_SNR'][()]
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
        
                ra = 2.0*np.pi*f1['ra'][()]
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.arcsin(1.0-2.0*f1['dec'][()])
                
#                ra = f1['ra'][()]
#                dec = f1['dec'][()]
                
                f1.close()
        
            elif(data == 'real_event'):
                f1 = h5py.File(data_config.BBH.path_test_real, 'r')
                
                h1_snr = f1['H1_SNR'][()]
                l1_snr = f1['L1_SNR'][()]
                v1_snr = f1['V1_SNR'][()]
                
                h1 = np.ones(len(h1_snr), dtype=bool)
                l1 = np.ones(len(l1_snr), dtype=bool)
                v1 = np.ones(len(v1_snr), dtype=bool)
        
                ra = 2.0*np.pi*f1['ra'][()]
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.arcsin(1.0-2.0*f1['dec'][()])

#                ra = f1['ra'][()]
#                dec = f1['dec'][()]
                
                f1.close()
                
        elif(self.dataset == 'BNS'):
            if((data == 'train') and (snr_range_train=='high')):
                f1 = h5py.File(data_config.BNS.path_train_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_2, 'r')
                
                h1_snr_1 = f1['H1_SNR'][0:22000][()]
                l1_snr_1 = f1['L1_SNR'][0:22000][()]
                v1_snr_1 = f1['V1_SNR'][0:22000][()]
                
                h1_snr_2 = f2['H1_SNR'][0:86000][()]
                l1_snr_2 = f2['L1_SNR'][0:86000][()]
                v1_snr_2 = f2['V1_SNR'][0:86000][()]
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2], axis=0)
                v1_snr = np.concatenate([v1_snr_1, v1_snr_2], axis=0)
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
        
                ra_1 = 2.0*np.pi*f1['ra'][0:22000][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][0:22000][()])

#                ra_1 = f1['ra'][0:22000][()]
#                dec_1 = f1['dec'][0:22000][()]
                
                ra_2 = 2.0*np.pi*f2['ra'][0:86000][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][0:86000][()])

#                ra_2 = f2['ra'][0:86000][()]
#                dec_2 = f2['dec'][0:86000][()]
                
                ra = np.concatenate([ra_1, ra_2], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.concatenate([dec_1, dec_2], axis=0)
                
                f1.close()
                f2.close()
                
            elif((data == 'train') and (snr_range_train=='low')):
                f1 = h5py.File(data_config.BNS.path_train_low_snr_1, 'r')
                f2 = h5py.File(data_config.BNS.path_train_low_snr_2, 'r')
                f3 = h5py.File(data_config.BNS.path_train_low_snr_3, 'r')
                
                h1_snr_1 = f1['H1_SNR'][0:12000][()]
                l1_snr_1 = f1['L1_SNR'][0:12000][()]
                v1_snr_1 = f1['V1_SNR'][0:12000][()]
                
                h1_snr_2 = f2['H1_SNR'][0:36000][()]
                l1_snr_2 = f2['L1_SNR'][0:36000][()]
                v1_snr_2 = f2['V1_SNR'][0:36000][()]
                
                h1_snr_3 = f3['H1_SNR'][0:52000][()]
                l1_snr_3 = f3['L1_SNR'][0:52000][()]
                v1_snr_3 = f3['V1_SNR'][0:52000][()]
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2, h1_snr_3], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2, l1_snr_3], axis=0)
                v1_snr = np.concatenate([v1_snr_1, v1_snr_2, l1_snr_3], axis=0)
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
        
                ra_1 = 2.0*np.pi*f1['ra'][0:12000][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][0:12000][()])
                
                ra_2 = 2.0*np.pi*f2['ra'][0:36000][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][0:36000][()])
                
                ra_3 = 2.0*np.pi*f3['ra'][0:52000][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][0:52000][()])
         
                ra = np.concatenate([ra_1, ra_2, ra_3], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.concatenate([dec_1, dec_2, dec_3], axis=0)
                
                f1.close()
                f2.close()
                f3.close()
                
            if((data == 'train_real') and (snr_range_train=='high')):
                f1 = h5py.File(data_config.BNS.path_train_real_2, 'r')
                f2 = h5py.File(data_config.BNS.path_train_real_3, 'r')
                
                h1_snr_1 = f1['H1_SNR'][0:42000][()]
                l1_snr_1 = f1['L1_SNR'][0:42000][()]
                v1_snr_1 = f1['V1_SNR'][0:42000][()]
                
                h1_snr_2 = f2['H1_SNR'][0:46000][()]
                l1_snr_2 = f2['L1_SNR'][0:46000][()]
                v1_snr_2 = f2['V1_SNR'][0:46000][()]
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2], axis=0)
                v1_snr = np.concatenate([v1_snr_1, v1_snr_2], axis=0)
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
        
                ra_1 = 2.0*np.pi*f1['ra'][0:42000][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][0:42000][()])

#                ra_1 = f1['ra'][0:22000][()]
#                dec_1 = f1['dec'][0:22000][()]
                
                ra_2 = 2.0*np.pi*f2['ra'][0:46000][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][0:46000][()])

#                ra_2 = f2['ra'][0:86000][()]
#                dec_2 = f2['dec'][0:86000][()]
                
                ra = np.concatenate([ra_1, ra_2], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.concatenate([dec_1, dec_2], axis=0)
                
                f1.close()
                f2.close()
                
                
            elif(data == 'test'):
                if(snr_range_test == 'high'):
                    f1 = h5py.File(data_config.BNS.path_test, 'r')
                elif(snr_range_test == 'low'):
                    f1 = h5py.File(data_config.BNS.path_test_low_SNR, 'r')
                
                h1_snr = f1['H1_SNR'][()]
                l1_snr = f1['L1_SNR'][()]   
                v1_snr = f1['V1_SNR'][()]  
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                v1 = v1_snr > self.min_snr
        
                ra = 2.0*np.pi*f1['ra'][()]
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.arcsin(1.0-2.0*f1['dec'][()])

#                ra = f1['ra'][()]
#                dec = f1['dec'][()]
                
                f1.close()
        
            elif(data == 'real_event'):
                f1 = h5py.File(data_config.BNS.path_test_GW170817, 'r')
                
                h1_snr = f1['H1_SNR'][()]
                l1_snr = f1['L1_SNR'][()]
                v1_snr = f1['V1_SNR'][()]
                
                h1 = np.ones(len(h1_snr), dtype=bool)
                l1 = np.ones(len(l1_snr), dtype=bool)
                v1 = np.ones(len(v1_snr), dtype=bool)
        
                ra = 2.0*np.pi*f1['ra'][()]
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.arcsin(1.0-2.0*f1['dec'][()])

#                ra = f1['ra'][()]
#                dec = f1['dec'][()]
                
                f1.close()
        
        index = np.zeros(num_samples, dtype = bool)
        
        for i in range(num_samples):
            if(self.num_det == 3):
                if(h1[i] == True and l1[i] == True and v1[i] == True):
                    index[i] = True
                    
#            elif(self.num_det == 2):
#                if(h1[i] == True and l1[i] == True):
#                    index[i] = True
                
        X_real = X_real[index == True]
        X_imag = X_imag[index == True]
        y = y[index == True]
        ra_x = ra_x[index == True]
        ra_y = ra_y[index == True]
        ra = ra[index == True]
        dec = dec[index == True]
        h1_snr = h1_snr[index==True]
        l1_snr = l1_snr[index==True]
        v1_snr = v1_snr[index==True]

        return X_real, X_imag, y, ra_x, ra_y, ra, dec, h1_snr, l1_snr, v1_snr
    
    #    @staticmethod
    def load_2_det_samples(self, data_config, X_real, X_imag, y, num_samples, snr_range_train, snr_range_test, data):
        """Loads 2 det samples and parameters from path"""
        if(self.dataset == 'BNS'):
            
            if((data == 'train') and (snr_range_train=='low')):
                
                f1 = h5py.File(data_config.BNS.path_train_2_det_low_SNR, 'r')
                f2 = h5py.File(data_config.BNS.path_train_low_snr_1, 'r')
                f3 = h5py.File(data_config.BNS.path_train_low_snr_2, 'r')
                f4 = h5py.File(data_config.BNS.path_train_low_snr_3, 'r')
                f5 = h5py.File(data_config.BNS.path_train_1, 'r')
                f6 = h5py.File(data_config.BNS.path_train_2, 'r')
                f7 = h5py.File(data_config.BNS.path_train_2_det_low_SNR_1, 'r')
                f8 = h5py.File(data_config.BNS.path_train_2_det_high_SNR_1, 'r')
                f9 = h5py.File(data_config.BNS.path_train_2_det_high_SNR_2, 'r')
                
                h1_snr_1 = f1['h1_snr'][()]
                l1_snr_1 = f1['l1_snr'][()]                
                h1_snr_2 = f2['H1_SNR'][0:12000][()]
                l1_snr_2 = f2['L1_SNR'][0:12000][()]
                h1_snr_3 = f3['H1_SNR'][0:36000][()]
                l1_snr_3 = f3['L1_SNR'][0:36000][()]
                h1_snr_4 = f4['H1_SNR'][0:52000][()]
                l1_snr_4 = f4['L1_SNR'][0:52000][()]
                h1_snr_5 = f5['H1_SNR'][()]
                l1_snr_5 = f5['L1_SNR'][()]
                h1_snr_6 = f6['H1_SNR'][()]
                l1_snr_6 = f6['L1_SNR'][()]
                h1_snr_7 = f7['H1_SNR'][()]
                l1_snr_7 = f7['L1_SNR'][()]
                h1_snr_8 = f8['H1_SNR'][()]
                l1_snr_8 = f8['L1_SNR'][()]
                h1_snr_9 = f9['H1_SNR'][()]
                l1_snr_9 = f9['L1_SNR'][()]
                
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2, h1_snr_3, h1_snr_4, h1_snr_5, h1_snr_6, h1_snr_7], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2, l1_snr_3, l1_snr_4, l1_snr_5, l1_snr_6, l1_snr_7], axis=0)
                
#                h1_snr = h1_snr[0:50000]
#                l1_snr = l1_snr[0:50000]
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
            
                network_snr = np.sqrt(h1_snr**2 + l1_snr**2)                
                net_snr = network_snr > 12
        
                ra_1 = 2.0*np.pi*f1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                ra_2 = 2.0*np.pi*f2['ra'][0:12000][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][0:12000][()])
                ra_3 = 2.0*np.pi*f3['ra'][0:36000][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][0:36000][()])
                ra_4 = 2.0*np.pi*f4['ra'][0:52000][()]
                dec_4 = np.arcsin(1.0-2.0*f4['dec'][0:52000][()])
                ra_5 = 2.0*np.pi*f5['ra'][()]
                dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                ra_6 = 2.0*np.pi*f6['ra'][()]
                dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                ra_7 = 2.0*np.pi*f7['ra'][()]
                dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                ra_8 = 2.0*np.pi*f8['ra'][()]
                dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                ra_9 = 2.0*np.pi*f9['ra'][()]
                dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])                
                
                ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7], axis=0)
                dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7], axis=0)
                
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
#                ra = ra[0:50000]
#                ra_x = ra_x[0:50000]
#                ra_y = ra_y[0:50000]
#                dec = dec[0:50000]
                
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
                f8.close()
                f9.close()
                
            elif((data == 'test') and (snr_range_test == 'low')):
                
                f1 = h5py.File(data_config.BNS.path_test_2_det_low_SNR, 'r')
                f2 = h5py.File(data_config.BNS.path_test, 'r')
                f3 = h5py.File(data_config.BNS.path_test_2_det_high_SNR, 'r')
                 
                h1_snr_1 = f1['H1_SNR'][()]
                l1_snr_1 = f1['L1_SNR'][()]
                
                h1_snr_2 = f2['H1_SNR'][()]
                l1_snr_2 = f2['L1_SNR'][()]
                
                h1_snr_3 = f3['H1_SNR'][()]
                l1_snr_3 = f3['L1_SNR'][()]
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2, h1_snr_3], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2, l1_snr_3], axis=0)
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                
                network_snr = np.sqrt(h1_snr**2 + l1_snr**2)
                net_snr = network_snr > 12
        
                ra_1 = 2.0*np.pi*f1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                
                ra_2 = 2.0*np.pi*f2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                
                ra_3 = 2.0*np.pi*f3['ra'][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                
                ra = np.concatenate([ra_1, ra_2, ra_3], axis=0)
                dec = np.concatenate([dec_1, dec_2, dec_3], axis=0)
                
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                f1.close()
                f2.close()
                f3.close()
                
        elif(self.dataset == 'NSBH'):
            
            if((data == 'train') and (snr_range_train=='low')):
                
                f1 = h5py.File(data_config.NSBH.path_train_1, 'r')
                f2 = h5py.File(data_config.NSBH.path_train_2, 'r')
                f3 = h5py.File(data_config.NSBH.path_train_3, 'r')
                f4 = h5py.File(data_config.NSBH.path_train_4, 'r')
                f5 = h5py.File(data_config.NSBH.path_train_low_snr_1, 'r')
                f6 = h5py.File(data_config.NSBH.path_train_low_snr_2, 'r')
                f7 = h5py.File(data_config.NSBH.path_train_low_snr_3, 'r')
                
                h1_snr_1 = f1['H1_SNR'][()]
                l1_snr_1 = f1['L1_SNR'][()]                
                h1_snr_2 = f2['H1_SNR'][0:30000][()]
                l1_snr_2 = f2['L1_SNR'][0:30000][()]
                h1_snr_3 = f3['H1_SNR'][()]
                l1_snr_3 = f3['L1_SNR'][()]
                h1_snr_4 = f4['H1_SNR'][()]
                l1_snr_4 = f4['L1_SNR'][()]
                h1_snr_5 = f5['H1_SNR'][()]
                l1_snr_5 = f5['L1_SNR'][()]
                h1_snr_6 = f6['H1_SNR'][()]
                l1_snr_6 = f6['L1_SNR'][()]
                h1_snr_7 = f7['H1_SNR'][()]
                l1_snr_7 = f7['L1_SNR'][()]
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2, h1_snr_3, h1_snr_4, h1_snr_5, h1_snr_6, h1_snr_7], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2, l1_snr_3, l1_snr_4, l1_snr_5, l1_snr_6, l1_snr_7], axis=0)
                
#                h1_snr = h1_snr[0:50000]
#                l1_snr = l1_snr[0:50000]
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
            
                network_snr = np.sqrt(h1_snr**2 + l1_snr**2)                
                net_snr = network_snr > 12
        
                ra_1 = 2.0*np.pi*f1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                ra_2 = 2.0*np.pi*f2['ra'][0:30000][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][0:30000][()])
                ra_3 = 2.0*np.pi*f3['ra'][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                ra_4 = 2.0*np.pi*f4['ra'][()]
                dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                ra_5 = 2.0*np.pi*f5['ra'][()]
                dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                ra_6 = 2.0*np.pi*f6['ra'][()]
                dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                ra_7 = 2.0*np.pi*f7['ra'][()]
                dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                
                ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7], axis=0)
                dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7], axis=0)
                
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
#                ra = ra[0:50000]
#                ra_x = ra_x[0:50000]
#                ra_y = ra_y[0:50000]
#                dec = dec[0:50000]
                
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
                
            elif((data == 'test') and (snr_range_test == 'low')):
                
                f_test_1 = h5py.File(data_config.NSBH.path_test, 'r')
                f_test_2 = h5py.File(data_config.NSBH.path_test_low_snr, 'r')
                             
                h1_snr_1 = f_test_1['H1_SNR'][()]
                l1_snr_1 = f_test_1['L1_SNR'][()]
                
                h1_snr_2 = f_test_2['H1_SNR'][()]
                l1_snr_2 = f_test_2['L1_SNR'][()]                 
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2], axis=0)
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                
                network_snr = np.sqrt(h1_snr**2 + l1_snr**2)
                net_snr = network_snr > 12
        
                ra_1 = 2.0*np.pi*f_test_1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
                
                ra_2 = 2.0*np.pi*f_test_2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
                
                ra = np.concatenate([ra_1, ra_2], axis=0)
                dec = np.concatenate([dec_1, dec_2], axis=0)
                
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                f_test_1.close()
                f_test_2.close()
                
        elif(self.dataset == 'BBH'):
            
            if((data == 'train') and (snr_range_train=='low')):
                
                f1 = h5py.File(data_config.BBH.path_train, 'r')
                f2 = h5py.File(data_config.BBH.path_train_low_SNR, 'r')
                
                h1_snr_1 = f1['H1_SNR'][2000:200000][()]
                l1_snr_1 = f1['L1_SNR'][2000:200000][()]                
                h1_snr_2 = f2['H1_SNR'][()]
                l1_snr_2 = f2['L1_SNR'][()]
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2], axis=0)
                
#                h1_snr = h1_snr[0:50000]
#                l1_snr = l1_snr[0:50000]
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
            
                network_snr = np.sqrt(h1_snr**2 + l1_snr**2)                
                net_snr = network_snr > 12
        
                ra_1 = 2.0*np.pi*f1['ra'][2000:200000][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][2000:200000][()])
                
                ra_2 = 2.0*np.pi*f2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                
                ra = np.concatenate([ra_1, ra_2], axis=0)
                dec = np.concatenate([dec_1, dec_2], axis=0)
                
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
#                ra = ra[0:50000]
#                ra_x = ra_x[0:50000]
#                ra_y = ra_y[0:50000]
#                dec = dec[0:50000]
                
                f1.close()
                f2.close()
                
            elif((data == 'test') and (snr_range_test == 'low')):
                
                f_test_1 = h5py.File(data_config.BBH.path_train, 'r')
                f_test_2 = h5py.File(data_config.BBH.path_test_low_SNR, 'r')
                             
                h1_snr_1 = f_test_1['H1_SNR'][()]
                l1_snr_1 = f_test_1['L1_SNR'][()]
                
                h1_snr_2 = f_test_2['H1_SNR'][()]
                l1_snr_2 = f_test_2['L1_SNR'][()]                 
                
                h1_snr = np.concatenate([h1_snr_1, h1_snr_2], axis=0)
                l1_snr = np.concatenate([l1_snr_1, l1_snr_2], axis=0)
                
                h1 = h1_snr > self.min_snr
                l1 = l1_snr > self.min_snr
                
                network_snr = np.sqrt(h1_snr**2 + l1_snr**2)
                net_snr = network_snr > 12
        
                ra_1 = 2.0*np.pi*f_test_1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
                
                ra_2 = 2.0*np.pi*f_test_2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
                
                ra = np.concatenate([ra_1, ra_2], axis=0)
                dec = np.concatenate([dec_1, dec_2], axis=0)
                
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                f_test_1.close()
                f_test_2.close()
        
        
        index = np.zeros(num_samples, dtype = bool)
        
        for i in range(num_samples):
            if(h1[i] == True and l1[i] == True and net_snr[i] == True):
                index[i] = True
                                  
        X_real = X_real[index == True]
        X_imag = X_imag[index == True]
        y = y[index == True]
        ra_x = ra_x[index == True]
        ra_y = ra_y[index == True]
        ra = ra[index == True]
        dec = dec[index == True]
        h1_snr = h1_snr[index==True]
        l1_snr = l1_snr[index==True]

        return X_real, X_imag, y, ra_x, ra_y, ra, dec, h1_snr, l1_snr
    
