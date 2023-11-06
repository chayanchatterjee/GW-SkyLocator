"""Data Loader"""

# Imports
import numpy as np
import h5py
import healpy as hp

class DataLoader:
    """Data Loader class"""        
    
    @staticmethod
    def load_train_3_det_data(data_config):
        """Loads dataset from path"""
        # NSBH
        if((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_train=='high') and (data_config.train.PSD == 'aLIGO')):
            f1 = h5py.File(data_config.data.NSBH.path_train_1, 'r')
            f2 = h5py.File(data_config.data.NSBH.path_train_2, 'r')
            f3 = h5py.File(data_config.data.NSBH.path_train_3, 'r')
            f4 = h5py.File(data_config.data.NSBH.path_train_4, 'r')
            
            h1_real_52k = np.real(f1['h1_snr_series'][()] )
            l1_real_52k = np.real(f1['l1_snr_series'][()] )
            v1_real_52k = np.real(f1['v1_snr_series'][()] )
        
            h1_real_30k = np.real(f2['h1_snr_series'][()] )
            l1_real_30k = np.real(f2['l1_snr_series'][()] )
            v1_real_30k = np.real(f2['v1_snr_series'][()] )
            
            h1_real_12k = np.real(f3['h1_snr_series'][()] )
            l1_real_12k = np.real(f3['l1_snr_series'][()] )
            v1_real_12k = np.real(f3['v1_snr_series'][()] )
        
            h1_real_6k = np.real(f4['h1_snr_series'][()] )
            l1_real_6k = np.real(f4['l1_snr_series'][()] )
            v1_real_6k = np.real(f4['v1_snr_series'][()] )
            
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
            
        elif((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_train=='low') and (data_config.train.PSD == 'aLIGO')):
            
            f1 = h5py.File(data_config.data.NSBH.path_train_low_snr_1, 'r')
            f2 = h5py.File(data_config.data.NSBH.path_train_low_snr_2, 'r')

            h1_real_60k = np.real(f1['h1_snr_series'][0:60000][()] )
            l1_real_60k = np.real(f1['l1_snr_series'][0:60000][()] )
            v1_real_60k = np.real(f1['v1_snr_series'][0:60000][()] )

            h1_imag_60k = np.imag((f1['h1_snr_series'][0:60000][()]))
            l1_imag_60k = np.imag((f1['l1_snr_series'][0:60000][()]))
            v1_imag_60k = np.imag((f1['v1_snr_series'][0:60000][()]))
            
            h1_real_40k = np.real(f2['h1_snr_series'][0:40000][()] )
            l1_real_40k = np.real(f2['l1_snr_series'][0:40000][()] )
            v1_real_40k = np.real(f2['v1_snr_series'][0:40000][()] )
    
            h1_imag_40k = np.imag((f2['h1_snr_series'][0:40000][()]))
            l1_imag_40k = np.imag((f2['l1_snr_series'][0:40000][()]))
            v1_imag_40k = np.imag((f2['v1_snr_series'][0:40000][()]))
                
            h1_real = np.concatenate([h1_real_60k, h1_real_40k], axis=0)
            l1_real = np.concatenate([l1_real_60k, l1_real_40k], axis=0)
            v1_real = np.concatenate([v1_real_60k, v1_real_40k], axis=0)
            
            h1_imag = np.concatenate([h1_imag_60k, h1_imag_40k], axis=0)
            l1_imag = np.concatenate([l1_imag_60k, l1_imag_40k], axis=0)
            v1_imag = np.concatenate([v1_imag_60k, v1_imag_40k], axis=0)
            
            f1.close()
            f2.close()
            
        elif((data_config.train.dataset == 'NSBH') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'design')):
                
            f1 = h5py.File(data_config.data.NSBH.path_train_design_1, 'r')
            f2 = h5py.File(data_config.data.NSBH.path_train_design_2, 'r')
            f3 = h5py.File(data_config.data.NSBH.path_train_design_3, 'r')
            f4 = h5py.File(data_config.data.NSBH.path_train_design_4, 'r')
            f5 = h5py.File(data_config.data.NSBH.path_train_design_5, 'r')
                
            f6 = h5py.File(data_config.data.NSBH.path_train_1, 'r')
            f7 = h5py.File(data_config.data.NSBH.path_train_2, 'r')
            f8 = h5py.File(data_config.data.NSBH.path_train_3, 'r')
            f9 = h5py.File(data_config.data.NSBH.path_train_4, 'r')
                
            f10 = h5py.File(data_config.data.NSBH.path_train_low_snr_1, 'r')
            f11 = h5py.File(data_config.data.NSBH.path_train_low_snr_2, 'r')
            
            h1_real_52k = np.real(f6['h1_snr_series'][()] )
            l1_real_52k = np.real(f6['l1_snr_series'][()] )
            v1_real_52k = np.real(f6['v1_snr_series'][()] )
        
            h1_real_30k = np.real(f7['h1_snr_series'][()] )
            l1_real_30k = np.real(f7['l1_snr_series'][()] )
            v1_real_30k = np.real(f7['v1_snr_series'][()] )
            
            h1_real_12k = np.real(f8['h1_snr_series'][()] )
            l1_real_12k = np.real(f8['l1_snr_series'][()] )
            v1_real_12k = np.real(f8['v1_snr_series'][()] )
        
            h1_real_6k = np.real(f9['h1_snr_series'][()] )
            l1_real_6k = np.real(f9['l1_snr_series'][()] )
            v1_real_6k = np.real(f9['v1_snr_series'][()] )
                
            h1_imag_52k = np.imag(f6['h1_snr_series'][()] )
            l1_imag_52k = np.imag(f6['l1_snr_series'][()] )
            v1_imag_52k = np.imag(f6['v1_snr_series'][()] )
        
            h1_imag_30k = np.imag(f7['h1_snr_series'][()] )
            l1_imag_30k = np.imag(f7['l1_snr_series'][()] )
            v1_imag_30k = np.imag(f7['v1_snr_series'][()] )
        
            h1_imag_12k = np.imag(f8['h1_snr_series'][()] )
            l1_imag_12k = np.imag(f8['l1_snr_series'][()] )
            v1_imag_12k = np.imag(f8['v1_snr_series'][()] )
        
            h1_imag_6k = np.imag(f9['h1_snr_series'][()] )
            l1_imag_6k = np.imag(f9['l1_snr_series'][()] )
            v1_imag_6k = np.imag(f9['v1_snr_series'][()] )
                
                
            h1_real_60k = np.real(f10['h1_snr_series'][0:60000][()] )
            l1_real_60k = np.real(f10['l1_snr_series'][0:60000][()] )
            v1_real_60k = np.real(f10['v1_snr_series'][0:60000][()] )

            h1_imag_60k = np.imag((f10['h1_snr_series'][0:60000][()]))
            l1_imag_60k = np.imag((f10['l1_snr_series'][0:60000][()]))
            v1_imag_60k = np.imag((f10['v1_snr_series'][0:60000][()]))
            
            h1_real_40k = np.real(f11['h1_snr_series'][0:40000][()] )
            l1_real_40k = np.real(f11['l1_snr_series'][0:40000][()] )
            v1_real_40k = np.real(f11['v1_snr_series'][0:40000][()] )
    
            h1_imag_40k = np.imag((f11['h1_snr_series'][0:40000][()]))
            l1_imag_40k = np.imag((f11['l1_snr_series'][0:40000][()]))
            v1_imag_40k = np.imag((f11['v1_snr_series'][0:40000][()]))                

            h1_real_1 = np.real(f1['h1_snr_series'][()] )
            l1_real_1 = np.real(f1['l1_snr_series'][()] )
            v1_real_1 = np.real(f1['v1_snr_series'][()] )

            h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
            l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
            v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
            h1_real_2 = np.real(f2['h1_snr_series'][()] )
            l1_real_2 = np.real(f2['l1_snr_series'][()] )
            v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
            h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
            v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
            h1_real_3 = np.real(f3['h1_snr_series'][()] )
            l1_real_3 = np.real(f3['l1_snr_series'][()] )
            v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
            h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
            l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
            v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
            h1_real_4 = np.real(f4['h1_snr_series'][()] )
            l1_real_4 = np.real(f4['l1_snr_series'][()] )
            v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
            h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
            l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
            v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
            h1_real_5 = np.real(f5['h1_snr_series'][()] )
            l1_real_5 = np.real(f5['l1_snr_series'][()] )
            v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
            h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
            l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
            v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
            
            
            h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_52k, h1_real_30k, h1_real_12k, h1_real_6k, h1_real_60k, h1_real_40k], axis=0)                
            l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_52k, l1_real_30k, l1_real_12k, l1_real_6k, l1_real_60k, l1_real_40k], axis=0)        
            v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_52k, v1_real_30k, v1_real_12k, v1_real_6k, v1_real_60k, v1_real_40k], axis=0)
            
            h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_52k, h1_imag_30k, h1_imag_12k, h1_imag_6k, h1_imag_60k, h1_imag_40k], axis=0)
            l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_52k, l1_imag_30k, l1_imag_12k, l1_imag_6k, l1_imag_60k, l1_imag_40k], axis=0)
            v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_52k, v1_imag_30k, v1_imag_12k, v1_imag_6k, v1_imag_60k, v1_imag_40k], axis=0)

            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
            f7.close()
            f8.close()
            f9.close()
            f10.close()
            f11.close()
        
        
        elif((data_config.train.dataset == 'NSBH') and (data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O2')):
            
            f1 = h5py.File(data_config.data.NSBH.path_train_real_noise_1, 'r')
            f2 = h5py.File(data_config.data.NSBH.path_train_real_noise_2, 'r')
            f3 = h5py.File(data_config.data.NSBH.path_train_real_noise_3, 'r')
            f4 = h5py.File(data_config.data.NSBH.path_train_real_noise_4, 'r')
            f5 = h5py.File(data_config.data.NSBH.path_train_real_noise_5, 'r')
            
            f6 = h5py.File(data_config.data.BBH.path_train_real_noise_1, 'r')
            f7 = h5py.File(data_config.data.BBH.path_train_real_noise_2, 'r')
            f8 = h5py.File(data_config.data.BBH.path_train_real_noise_3, 'r')
            f9 = h5py.File(data_config.data.BBH.path_train_real_noise_4, 'r')
            f10 = h5py.File(data_config.data.BBH.path_train_real_noise_5, 'r')
                
            h1_real_1 = np.real(f1['h1_snr_series'][()] )
            l1_real_1 = np.real(f1['l1_snr_series'][()] )
            v1_real_1 = np.real(f1['v1_snr_series'][()] )

            h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
            l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
            v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
            h1_real_2 = np.real(f2['h1_snr_series'][()] )
            l1_real_2 = np.real(f2['l1_snr_series'][()] )
            v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
            h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
            v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
            h1_real_3 = np.real(f3['h1_snr_series'][()] )
            l1_real_3 = np.real(f3['l1_snr_series'][()] )
            v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
            h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
            l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
            v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
            h1_real_4 = np.real(f4['h1_snr_series'][()] )
            l1_real_4 = np.real(f4['l1_snr_series'][()] )
            v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
            h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
            l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
            v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
            h1_real_5 = np.real(f5['h1_snr_series'][()] )
            l1_real_5 = np.real(f5['l1_snr_series'][()] )
            v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
            h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
            l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
            v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
            
                
            h1_real_6 = np.real(f6['h1_snr_series'][()] )
            l1_real_6 = np.real(f6['l1_snr_series'][()] )
            v1_real_6 = np.real(f6['v1_snr_series'][()] )

            h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
            l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
            v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
            
            h1_real_7 = np.real(f7['h1_snr_series'][()] )
            l1_real_7 = np.real(f7['l1_snr_series'][()] )
            v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
            h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
            l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
            v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
            h1_real_8 = np.real(f8['h1_snr_series'][()] )
            l1_real_8 = np.real(f8['l1_snr_series'][()] )
            v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
            h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
            l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
            v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
            h1_real_9 = np.real(f9['h1_snr_series'][()] )
            l1_real_9 = np.real(f9['l1_snr_series'][()] )
            v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
            h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
            l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
            v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
                
            h1_real_10 = np.real(f10['h1_snr_series'][()] )
            l1_real_10 = np.real(f10['l1_snr_series'][()] )
            v1_real_10 = np.real(f10['v1_snr_series'][()] )
    
            h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
            l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
            v1_imag_10 = np.imag((f10['v1_snr_series'][()]))
            
            
            h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9, h1_real_10], axis=0) 
            l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9, l1_real_10], axis=0) 
            v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9, v1_real_10], axis=0)
            
            h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9, h1_imag_10], axis=0)
            l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9, l1_imag_10], axis=0)
            v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9, v1_imag_10], axis=0)
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
            f7.close()
            f8.close()
            f9.close()
            f10.close()
        
            
        # BBH
        elif((data_config.train.dataset == 'BBH') and (data_config.train.snr_range_train=='high') and (data_config.train.PSD == 'aLIGO')):
            
            f1 = h5py.File(data_config.data.BBH.path_train, 'r')

            h1_real = np.real(f1['h1_snr_series'][0:100000][()] )
            l1_real = np.real(f1['l1_snr_series'][0:100000][()] )
            v1_real = np.real(f1['v1_snr_series'][0:100000][()] )

            h1_imag = np.imag(f1['h1_snr_series'][0:100000][()] )
            l1_imag = np.imag(f1['l1_snr_series'][0:100000][()] )
            v1_imag = np.imag(f1['v1_snr_series'][0:100000][()] )
    
            f1.close()
        
        elif((data_config.train.dataset == 'BBH') and (data_config.train.snr_range_train=='low') and (data_config.train.PSD == 'aLIGO')):
            
            f1 = h5py.File(data_config.data.BBH.path_train_low_SNR, 'r')

            h1_real = np.real(f1['h1_snr_series'][0:100000][()] )
            l1_real = np.real(f1['l1_snr_series'][0:100000][()] )
            v1_real = np.real(f1['v1_snr_series'][0:100000][()] )

            h1_imag = np.imag(f1['h1_snr_series'][0:100000][()] )
            l1_imag = np.imag(f1['l1_snr_series'][0:100000][()] )
            v1_imag = np.imag(f1['v1_snr_series'][0:100000][()] )
    
            f1.close()
        
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'design')):
                
            f1 = h5py.File(data_config.data.BBH.path_train_design_1, 'r')
            f2 = h5py.File(data_config.data.BBH.path_train_design_2, 'r')
            f3 = h5py.File(data_config.data.BBH.path_train_design_3, 'r')
            f4 = h5py.File(data_config.data.BBH.path_train_design_4, 'r')
            f5 = h5py.File(data_config.data.BBH.path_train_design_5, 'r')
                
            f6 = h5py.File(data_config.data.BBH.path_train, 'r')
            f7 = h5py.File(data_config.data.BBH.path_train_low_SNR, 'r')
                
            h1_real_1 = np.real(f1['h1_snr_series'][()] )
            l1_real_1 = np.real(f1['l1_snr_series'][()] )
            v1_real_1 = np.real(f1['v1_snr_series'][()] )

            h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
            l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
            v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
            h1_real_2 = np.real(f2['h1_snr_series'][()] )
            l1_real_2 = np.real(f2['l1_snr_series'][()] )
            v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
            h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
            v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
            h1_real_3 = np.real(f3['h1_snr_series'][()] )
            l1_real_3 = np.real(f3['l1_snr_series'][()] )
            v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
            h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
            l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
            v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
            h1_real_4 = np.real(f4['h1_snr_series'][()] )
            l1_real_4 = np.real(f4['l1_snr_series'][()] )
            v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
            h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
            l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
            v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
            h1_real_5 = np.real(f5['h1_snr_series'][()] )
            l1_real_5 = np.real(f5['l1_snr_series'][()] )
            v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
            h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
            l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
            v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
                
            h1_real_6 = np.real(f6['h1_snr_series'][()] )
            l1_real_6 = np.real(f6['l1_snr_series'][()] )
            v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
            h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
            l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
            v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
                
            h1_real_7 = np.real(f7['h1_snr_series'][()] )
            l1_real_7 = np.real(f7['l1_snr_series'][()] )
            v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
            h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
            l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
            v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
            

#            h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_6, h1_real_7], axis=0) 
#            l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_6, l1_real_7], axis=0) 
#            v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_6, v1_real_7], axis=0)
            
#            h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_6, h1_imag_7], axis=0)
#            l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_6, l1_imag_7], axis=0)
#            v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_6, v1_imag_7], axis=0)
            
            h1_real = h1_real_6
            l1_real = l1_real_6
            v1_real = v1_real_6
            
            h1_imag = h1_imag_6
            l1_imag = l1_imag_6
            v1_imag = v1_imag_6
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
            f7.close()
        
        
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O2')):
            
            f1 = h5py.File(data_config.data.BBH.path_train_real_noise_1, 'r')
            f2 = h5py.File(data_config.data.BBH.path_train_real_noise_2, 'r')
            f3 = h5py.File(data_config.data.BBH.path_train_real_noise_3, 'r')
            f4 = h5py.File(data_config.data.BBH.path_train_real_noise_4, 'r')
            f5 = h5py.File(data_config.data.BBH.path_train_real_noise_5, 'r')
            f6 = h5py.File(data_config.data.BBH.path_train_real_noise_6, 'r')
            f7 = h5py.File(data_config.data.BBH.path_train_real_noise_7, 'r')
            f8 = h5py.File(data_config.data.BBH.path_train_real_noise_8, 'r')
            f9 = h5py.File(data_config.data.BBH.path_train_real_noise_9, 'r')
                
            h1_real_1 = np.real(f1['h1_snr_series'][()] )
            l1_real_1 = np.real(f1['l1_snr_series'][()] )
            v1_real_1 = np.real(f1['v1_snr_series'][()] )

            h1_imag_1 = np.real(np.imag((f1['h1_snr_series'][()])))
            l1_imag_1 = np.real(np.imag((f1['l1_snr_series'][()])))
            v1_imag_1 = np.real(np.imag((f1['v1_snr_series'][()])))
            
            h1_real_2 = np.real(f2['h1_snr_series'][()] )
            l1_real_2 = np.real(f2['l1_snr_series'][()] )
            v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
            h1_imag_2 = np.real(np.imag((f2['h1_snr_series'][()])))
            l1_imag_2 = np.real(np.imag((f2['l1_snr_series'][()])))
            v1_imag_2 = np.real(np.imag((f2['v1_snr_series'][()])))
                
            h1_real_3 = np.real(f3['h1_snr_series'][()] )
            l1_real_3 = np.real(f3['l1_snr_series'][()] )
            v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
            h1_imag_3 = np.real(np.imag((f3['h1_snr_series'][()])))
            l1_imag_3 = np.real(np.imag((f3['l1_snr_series'][()])))
            v1_imag_3 = np.real(np.imag((f3['v1_snr_series'][()])))
                
            h1_real_4 = np.real(f4['h1_snr_series'][()] )
            l1_real_4 = np.real(f4['l1_snr_series'][()] )
            v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
            h1_imag_4 = np.real(np.imag((f4['h1_snr_series'][()])))
            l1_imag_4 = np.real(np.imag((f4['l1_snr_series'][()])))
            v1_imag_4 = np.real(np.imag((f4['v1_snr_series'][()])))
                
            h1_real_5 = np.real(f5['h1_snr_series'][()] )
            l1_real_5 = np.real(f5['l1_snr_series'][()] )
            v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
            h1_imag_5 = np.real(np.imag((f5['h1_snr_series'][()])))
            l1_imag_5 = np.real(np.imag((f5['l1_snr_series'][()])))
            v1_imag_5 = np.real(np.imag((f5['v1_snr_series'][()])))
            
            h1_real_6 = np.real(f6['h1_snr_series'][()] )
            l1_real_6 = np.real(f6['l1_snr_series'][()] )
            v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
            h1_imag_6 = np.real(np.imag((f6['h1_snr_series'][()])))
            l1_imag_6 = np.real(np.imag((f6['l1_snr_series'][()])))
            v1_imag_6 = np.real(np.imag((f6['v1_snr_series'][()])))
                
            h1_real_7 = np.real(f7['h1_snr_series'][()] )
            l1_real_7 = np.real(f7['l1_snr_series'][()] )
            v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
            h1_imag_7 = np.real(np.imag((f7['h1_snr_series'][()])))
            l1_imag_7 = np.real(np.imag((f7['l1_snr_series'][()])))
            v1_imag_7 = np.real(np.imag((f7['v1_snr_series'][()])))
            
            h1_real_8 = np.real(f8['h1_snr_series'][()] )
            l1_real_8 = np.real(f8['l1_snr_series'][()] )
            v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
            h1_imag_8 = np.real(np.imag((f8['h1_snr_series'][()])))
            l1_imag_8 = np.real(np.imag((f8['l1_snr_series'][()])))
            v1_imag_8 = np.real(np.imag((f8['v1_snr_series'][()])))
                
            h1_real_9 = np.real(f9['h1_snr_series'][()] )
            l1_real_9 = np.real(f9['l1_snr_series'][()] )
            v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
            h1_imag_9 = np.real(np.imag((f9['h1_snr_series'][()])))
            l1_imag_9 = np.real(np.imag((f9['l1_snr_series'][()])))
            v1_imag_9 = np.real(np.imag((f9['v1_snr_series'][()])))
            
            
            h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9], axis=0) 
            l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9], axis=0) 
            v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9], axis=0)
            
            h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9], axis=0)
            l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9], axis=0)
            v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9], axis=0)
                      
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
            f7.close()
            f8.close()
            f9.close()

           
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O3')):
            
            f1 = h5py.File(data_config.data.BBH.path_train_O3_noise_1, 'r')
            f2 = h5py.File(data_config.data.BBH.path_train_O3_noise_2, 'r')
            f3 = h5py.File(data_config.data.BBH.path_train_O3_noise_3, 'r')
            f4 = h5py.File(data_config.data.BBH.path_train_O3_noise_4, 'r')
            
            h1_real_1 = np.real(f1['h1_snr_series'][()] )
            l1_real_1 = np.real(f1['l1_snr_series'][()] )
            v1_real_1 = np.real(f1['v1_snr_series'][()] )

            h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
            l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
            v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
            h1_real_2 = np.real(f2['h1_snr_series'][()] )
            l1_real_2 = np.real(f2['l1_snr_series'][()] )
            v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
            h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
            v1_imag_2 = np.imag((f2['v1_snr_series'][()]))    
            
            
            h1_real_3 = np.real(f3['h1_snr_series'][()] )
            l1_real_3 = np.real(f3['l1_snr_series'][()] )
            v1_real_3 = np.real(f3['v1_snr_series'][()] )

            h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
            l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
            v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
            
            h1_real_4 = np.real(f4['h1_snr_series'][()] )
            l1_real_4 = np.real(f4['l1_snr_series'][()] )
            v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
            h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
            l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
            v1_imag_4 = np.imag((f4['v1_snr_series'][()]))    
            
            
            h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4], axis=0) 
            l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4], axis=0) 
            v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4], axis=0)
            
            h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4], axis=0)
            l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4], axis=0)
            v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4], axis=0)
            
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            
        # BNS
        elif(data_config.train.dataset == 'BNS'):
            if((data_config.train.snr_range_train == 'high') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and ((data_config.train.PSD == 'aLIGO') or (data_config.train.PSD == 'design'))):
                f1 = h5py.File(data_config.data.BNS.path_train_1, 'r')
                f2 = h5py.File(data_config.data.BNS.path_train_2, 'r')

                h1_real_22k = np.real(f1['h1_snr_series'][0:22000][()] )
                l1_real_22k = np.real(f1['l1_snr_series'][0:22000][()] )
                v1_real_22k = np.real(f1['v1_snr_series'][0:22000][()] )

                h1_imag_22k = np.imag((f1['h1_snr_series'][0:22000][()]))
                l1_imag_22k = np.imag((f1['l1_snr_series'][0:22000][()]))
                v1_imag_22k = np.imag((f1['v1_snr_series'][0:22000][()]))
            
                h1_real_86k = np.real(f2['h1_snr_series'][0:86000][()] )
                l1_real_86k = np.real(f2['l1_snr_series'][0:86000][()] )
                v1_real_86k = np.real(f2['v1_snr_series'][0:86000][()] )
    
                h1_imag_86k = np.imag((f2['h1_snr_series'][0:86000][()]))
                l1_imag_86k = np.imag((f2['l1_snr_series'][0:86000][()]))
                v1_imag_86k = np.imag((f2['v1_snr_series'][0:86000][()]))
            
                h1_real = np.concatenate([h1_real_22k, h1_real_86k], axis=0)
                l1_real = np.concatenate([l1_real_22k, l1_real_86k], axis=0)
                v1_real = np.concatenate([v1_real_22k, v1_real_86k], axis=0)
            
                h1_imag = np.concatenate([h1_imag_22k, h1_imag_86k], axis=0)
                l1_imag = np.concatenate([l1_imag_22k, l1_imag_86k], axis=0)
                v1_imag = np.concatenate([v1_imag_22k, v1_imag_86k], axis=0)
            
                f1.close()
                f2.close()
                
                
            elif((data_config.train.snr_range_train == 'low') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and ((data_config.train.PSD == 'aLIGO'))):
                
                f1 = h5py.File(data_config.data.BNS.path_train_low_snr_1, 'r')
                f2 = h5py.File(data_config.data.BNS.path_train_low_snr_2, 'r')
                f3 = h5py.File(data_config.data.BNS.path_train_low_snr_3, 'r')

                h1_real_12k = np.real(f1['h1_snr_series'][0:12000][()] )
                l1_real_12k = np.real(f1['l1_snr_series'][0:12000][()] )
                v1_real_12k = np.real(f1['v1_snr_series'][0:12000][()] )

                h1_imag_12k = np.imag(f1['h1_snr_series'][0:12000][()] )
                l1_imag_12k = np.imag(f1['l1_snr_series'][0:12000][()] )
                v1_imag_12k = np.imag(f1['v1_snr_series'][0:12000][()] )
            
                h1_real_36k = np.real(f2['h1_snr_series'][0:36000][()] )
                l1_real_36k = np.real(f2['l1_snr_series'][0:36000][()] )
                v1_real_36k = np.real(f2['v1_snr_series'][0:36000][()] )
    
                h1_imag_36k = np.imag(f2['h1_snr_series'][0:36000][()] )
                l1_imag_36k = np.imag(f2['l1_snr_series'][0:36000][()] )
                v1_imag_36k = np.imag(f2['v1_snr_series'][0:36000][()] )
                
                h1_real_52k = np.real(f3['h1_snr_series'][0:52000][()] )
                l1_real_52k = np.real(f3['l1_snr_series'][0:52000][()] )
                v1_real_52k = np.real(f3['v1_snr_series'][0:52000][()] )
    
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
        
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'design')):

                f1 = h5py.File(data_config.data.BNS.path_train_design_1, 'r')
                f2 = h5py.File(data_config.data.BNS.path_train_design_2, 'r')
                f3 = h5py.File(data_config.data.BNS.path_train_design_3, 'r')
                f4 = h5py.File(data_config.data.BNS.path_train_design_4, 'r')
                f5 = h5py.File(data_config.data.BNS.path_train_design_5, 'r')
                f6 = h5py.File(data_config.data.BNS.path_train_design_6, 'r')
                f7 = h5py.File(data_config.data.BNS.path_train_design_7, 'r')
                f8 = h5py.File(data_config.data.BNS.path_train_design_8, 'r')
                f9 = h5py.File(data_config.data.BNS.path_train_design_9, 'r')
                f10 = h5py.File(data_config.data.BNS.path_train_design_10, 'r')
                   
                f11 = h5py.File(data_config.data.BNS.path_train_1, 'r')
                f12 = h5py.File(data_config.data.BNS.path_train_2, 'r')
                    
                f13 = h5py.File(data_config.data.BNS.path_train_design_high_SNR_1, 'r')
                f14 = h5py.File(data_config.data.BNS.path_train_design_high_SNR_2, 'r')

                h1_real_11 = np.real(f11['h1_snr_series'][()] )
                l1_real_11 = np.real(f11['l1_snr_series'][()] )
                v1_real_11 = np.real(f11['v1_snr_series'][()] )

                h1_imag_11 = np.imag((f11['h1_snr_series'][()]))
                l1_imag_11 = np.imag((f11['l1_snr_series'][()]))
                v1_imag_11 = np.imag((f11['v1_snr_series'][()]))
            
                h1_real_12 = np.real(f12['h1_snr_series'][()] )
                l1_real_12 = np.real(f12['l1_snr_series'][()] )
                v1_real_12 = np.real(f12['v1_snr_series'][()] )
    
                h1_imag_12 = np.imag((f12['h1_snr_series'][()]))
                l1_imag_12 = np.imag((f12['l1_snr_series'][()]))
                v1_imag_12 = np.imag((f12['v1_snr_series'][()]))

                h1_real_1 = np.real(f1['h1_snr_series'][()] )
                l1_real_1 = np.real(f1['l1_snr_series'][()] )
                v1_real_1 = np.real(f1['v1_snr_series'][()] )

                h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
                h1_real_2 = np.real(f2['h1_snr_series'][()] )
                l1_real_2 = np.real(f2['l1_snr_series'][()] )
                v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
                h1_real_3 = np.real(f3['h1_snr_series'][()] )
                l1_real_3 = np.real(f3['l1_snr_series'][()] )
                v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
                h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
                h1_real_4 = np.real(f4['h1_snr_series'][()] )
                l1_real_4 = np.real(f4['l1_snr_series'][()] )
                v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
                h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
                h1_real_5 = np.real(f5['h1_snr_series'][()] )
                l1_real_5 = np.real(f5['l1_snr_series'][()] )
                v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
                h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
                l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
                v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
                
                h1_real_6 = np.real(f6['h1_snr_series'][()] )
                l1_real_6 = np.real(f6['l1_snr_series'][()] )
                v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
                h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
                l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
                v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
                
                h1_real_7 = np.real(f7['h1_snr_series'][()] )
                l1_real_7 = np.real(f7['l1_snr_series'][()] )
                v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
                h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
                l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
                v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
                h1_real_8 = np.real(f8['h1_snr_series'][()] )
                l1_real_8 = np.real(f8['l1_snr_series'][()] )
                v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
                h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
                v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
                h1_real_9 = np.real(f9['h1_snr_series'][()] )
                l1_real_9 = np.real(f9['l1_snr_series'][()] )
                v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
                h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
                v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
                
                h1_real_10 = np.real(f10['h1_snr_series'][()] )
                l1_real_10 = np.real(f10['l1_snr_series'][()] )
                v1_real_10 = np.real(f10['v1_snr_series'][()] )
    
                h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
                l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
                v1_imag_10 = np.imag((f10['v1_snr_series'][()]))
                
                h1_real_13 = np.real(f13['h1_snr_series'][()] )
                l1_real_13 = np.real(f13['l1_snr_series'][()] )
                v1_real_13 = np.real(f13['v1_snr_series'][()] )
    
                h1_imag_13 = np.imag((f13['h1_snr_series'][()]))
                l1_imag_13 = np.imag((f13['l1_snr_series'][()]))
                v1_imag_13 = np.imag((f13['v1_snr_series'][()]))
                
                h1_real_14 = np.real(f14['h1_snr_series'][()] )
                l1_real_14 = np.real(f14['l1_snr_series'][()] )
                v1_real_14 = np.real(f14['v1_snr_series'][()] )
    
                h1_imag_14 = np.imag((f14['h1_snr_series'][()]))
                l1_imag_14 = np.imag((f14['l1_snr_series'][()]))
                v1_imag_14 = np.imag((f14['v1_snr_series'][()]))
                
                
                h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9, h1_real_10, h1_real_11, h1_real_12], axis=0)
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9, h1_imag_10, h1_imag_11, h1_imag_12], axis=0)
                        
                l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9, l1_real_10, l1_real_11, l1_real_12], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9, l1_imag_10, l1_imag_11, l1_imag_12], axis=0)
                        
                v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9, v1_real_10, v1_real_11, v1_real_12], axis=0)
                v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9, v1_imag_10, v1_imag_11, v1_imag_12], axis=0)
                       
            
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
                f8.close()
                f9.close()
                f10.close()
                f11.close()
                f12.close()
                f13.close()
                f14.close()
            
            
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == True) and (data_config.train.PSD == 'aLIGO')):
                
                if(data_config.train.train_negative_latency_seconds == '5'):
                
                    f1 = h5py.File(data_config.data.BNS.path_train_5_sec, 'r')

                    h1_real = np.real(f1['h1_snr_series'][()] )
                    l1_real = np.real(f1['l1_snr_series'][()] )
                    v1_real = np.real(f1['v1_snr_series'][()] )

                    h1_imag = np.realnp.imag((f1['h1_snr_series'][()]))
                    l1_imag = np.realnp.imag((f1['l1_snr_series'][()]))
                    v1_imag = np.realnp.imag((f1['v1_snr_series'][()]))
            
                    f1.close()
                
                elif(data_config.train.train_negative_latency_seconds == '10'):
                
                    f1 = h5py.File(data_config.data.BNS.path_train_10_sec, 'r')

                    h1_real = np.real(f1['h1_snr_series'][()] )
                    l1_real = np.real(f1['l1_snr_series'][()] )
                    v1_real = np.real(f1['v1_snr_series'][()] )

                    h1_imag = np.realnp.imag((f1['h1_snr_series'][()]))
                    l1_imag = np.realnp.imag((f1['l1_snr_series'][()]))
                    v1_imag = np.realnp.imag((f1['v1_snr_series'][()]))
            
                    f1.close()
                

            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == True) and ((data_config.train.PSD == 'design'))):
                
                if(data_config.train.train_negative_latency_seconds == '0'):
                    
                    f1 = h5py.File(data_config.data.BNS.path_train_design_1, 'r')
                    f2 = h5py.File(data_config.data.BNS.path_train_design_2, 'r')
                    f3 = h5py.File(data_config.data.BNS.path_train_design_3, 'r')
                    f4 = h5py.File(data_config.data.BNS.path_train_design_4, 'r')
                    f5 = h5py.File(data_config.data.BNS.path_train_design_5, 'r')
                    f6 = h5py.File(data_config.data.BNS.path_train_design_6, 'r')
                    f7 = h5py.File(data_config.data.BNS.path_train_design_7, 'r')
                    f8 = h5py.File(data_config.data.BNS.path_train_design_8, 'r')
                    f9 = h5py.File(data_config.data.BNS.path_train_design_9, 'r')
                    f10 = h5py.File(data_config.data.BNS.path_train_design_10, 'r')
                    
                    f11 = h5py.File(data_config.data.BNS.path_train_1, 'r')
                    f12 = h5py.File(data_config.data.BNS.path_train_2, 'r')
                    
                    f13 = h5py.File(data_config.data.BNS.path_train_design_high_SNR_1, 'r')
                    f14 = h5py.File(data_config.data.BNS.path_train_design_high_SNR_2, 'r')

                    h1_real_11 = np.real(f11['h1_snr_series'][()] )
                    l1_real_11 = np.real(f11['l1_snr_series'][()] )
                    v1_real_11 = np.real(f11['v1_snr_series'][()] )

                    h1_imag_11 = np.imag((f11['h1_snr_series'][()]))
                    l1_imag_11 = np.imag((f11['l1_snr_series'][()]))
                    v1_imag_11 = np.imag((f11['v1_snr_series'][()]))
            
                    h1_real_12 = np.real(f12['h1_snr_series'][()] )
                    l1_real_12 = np.real(f12['l1_snr_series'][()] )
                    v1_real_12 = np.real(f12['v1_snr_series'][()] )
    
                    h1_imag_12 = np.imag((f12['h1_snr_series'][()]))
                    l1_imag_12 = np.imag((f12['l1_snr_series'][()]))
                    v1_imag_12 = np.imag((f12['v1_snr_series'][()]))

                    h1_real_1 = np.real(f1['h1_snr_series'][()] )
                    l1_real_1 = np.real(f1['l1_snr_series'][()] )
                    v1_real_1 = np.real(f1['v1_snr_series'][()] )

                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
                    h1_real_2 = np.real(f2['h1_snr_series'][()] )
                    l1_real_2 = np.real(f2['l1_snr_series'][()] )
                    v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                    h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
                    h1_real_3 = np.real(f3['h1_snr_series'][()] )
                    l1_real_3 = np.real(f3['l1_snr_series'][()] )
                    v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
                    h1_real_4 = np.real(f4['h1_snr_series'][()] )
                    l1_real_4 = np.real(f4['l1_snr_series'][()] )
                    v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
                    h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                    l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                    v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
                    h1_real_5 = np.real(f5['h1_snr_series'][()] )
                    l1_real_5 = np.real(f5['l1_snr_series'][()] )
                    v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
                    h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
                    l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
                    v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
                
                    h1_real_6 = np.real(f6['h1_snr_series'][()] )
                    l1_real_6 = np.real(f6['l1_snr_series'][()] )
                    v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
                    h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
                    l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
                    v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
                
                    h1_real_7 = np.real(f7['h1_snr_series'][()] )
                    l1_real_7 = np.real(f7['l1_snr_series'][()] )
                    v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
                    h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
                    l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
                    v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
                    h1_real_8 = np.real(f8['h1_snr_series'][()] )
                    l1_real_8 = np.real(f8['l1_snr_series'][()] )
                    v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
                    h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                    l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
                    v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
                    h1_real_9 = np.real(f9['h1_snr_series'][()] )
                    l1_real_9 = np.real(f9['l1_snr_series'][()] )
                    v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
                    h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                    l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
                    v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
                
                    h1_real_10 = np.real(f10['h1_snr_series'][()] )
                    l1_real_10 = np.real(f10['l1_snr_series'][()] )
                    v1_real_10 = np.real(f10['v1_snr_series'][()] )
    
                    h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
                    l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
                    v1_imag_10 = np.imag((f10['v1_snr_series'][()]))
                
                    h1_real_13 = np.real(f13['h1_snr_series'][()] )
                    l1_real_13 = np.real(f13['l1_snr_series'][()] )
                    v1_real_13 = np.real(f13['v1_snr_series'][()] )
    
                    h1_imag_13 = np.imag((f13['h1_snr_series'][()]))
                    l1_imag_13 = np.imag((f13['l1_snr_series'][()]))
                    v1_imag_13 = np.imag((f13['v1_snr_series'][()]))
                
                    h1_real_14 = np.real(f14['h1_snr_series'][()] )
                    l1_real_14 = np.real(f14['l1_snr_series'][()] )
                    v1_real_14 = np.real(f14['v1_snr_series'][()] )
    
                    h1_imag_14 = np.imag((f14['h1_snr_series'][()]))
                    l1_imag_14 = np.imag((f14['l1_snr_series'][()]))
                    v1_imag_14 = np.imag((f14['v1_snr_series'][()]))
                
                
                    h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9, h1_real_10, h1_real_11, h1_real_12], axis=0)
                    h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9, h1_imag_10, h1_imag_11, h1_imag_12], axis=0)
                        
                    l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9, l1_real_10, l1_real_11, l1_real_12], axis=0)
                    l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9, l1_imag_10, l1_imag_11, l1_imag_12], axis=0)
                        
                    v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9, v1_real_10, v1_real_11, v1_real_12], axis=0)
                    v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9, v1_imag_10, v1_imag_11, v1_imag_12], axis=0)
                       

            
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                    f13.close()
                    f14.close()
                
                
                elif(data_config.train.train_negative_latency_seconds == '10'):
            
                    f1 = h5py.File(data_config.data.BNS.path_train_design_10_sec_1, 'r')
                    f2 = h5py.File(data_config.data.BNS.path_train_design_10_sec_2, 'r')
                    f3 = h5py.File(data_config.data.BNS.path_train_design_10_sec_3, 'r')
                    f4 = h5py.File(data_config.data.BNS.path_train_design_10_sec_4, 'r')
                    f5 = h5py.File(data_config.data.BNS.path_train_design_10_sec_5, 'r')
                    f6 = h5py.File(data_config.data.BNS.path_train_design_10_sec_6, 'r')
                    f7 = h5py.File(data_config.data.BNS.path_train_design_10_sec_7, 'r')
                    f8 = h5py.File(data_config.data.BNS.path_train_design_10_sec_8, 'r')
                    f9 = h5py.File(data_config.data.BNS.path_train_design_10_sec_9, 'r')
                    f10 = h5py.File(data_config.data.BNS.path_train_design_10_sec_10, 'r')
                    f11 = h5py.File(data_config.data.BNS.path_train_design_10_sec_11, 'r')
                    f12 = h5py.File(data_config.data.BNS.path_train_design_10_sec_12, 'r')
                    f13 = h5py.File(data_config.data.BNS.path_train_design_10_sec_13, 'r')
                    f14 = h5py.File(data_config.data.BNS.path_train_design_10_sec_14, 'r')
                    f15 = h5py.File(data_config.data.BNS.path_train_design_10_sec_15, 'r')
                    f16 = h5py.File(data_config.data.BNS.path_train_design_10_sec_16, 'r')
                    f17 = h5py.File(data_config.data.BNS.path_train_design_10_sec_17, 'r')
                    f18 = h5py.File(data_config.data.BNS.path_train_design_10_sec_18, 'r')
                    f19 = h5py.File(data_config.data.BNS.path_train_design_10_sec_19, 'r')
                    f20 = h5py.File(data_config.data.BNS.path_train_design_10_sec_20, 'r')

                    h1_real_1 = np.real(f1['h1_snr_series'][0:50000][()] )
                    l1_real_1 = np.real(f1['l1_snr_series'][0:50000][()] )
                    v1_real_1 = np.real(f1['v1_snr_series'][0:50000][()] )

                    h1_imag_1 = np.imag((f1['h1_snr_series'][0:50000][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][0:50000][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][0:50000][()]))
            
                    h1_real_2 = np.real(f2['h1_snr_series'][0:50000][()] )
                    l1_real_2 = np.real(f2['l1_snr_series'][0:50000][()] )
                    v1_real_2 = np.real(f2['v1_snr_series'][0:50000][()] )
    
                    h1_imag_2 = np.imag((f2['h1_snr_series'][0:50000][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][0:50000][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][0:50000][()]))
                
                    h1_real_3 = np.real(f3['h1_snr_series'][0:50000][()] )
                    l1_real_3 = np.real(f3['l1_snr_series'][0:50000][()] )
                    v1_real_3 = np.real(f3['v1_snr_series'][0:50000][()] )
    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][0:50000][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][0:50000][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][0:50000][()]))
                
                    h1_real_4 = np.real(f4['h1_snr_series'][()] )
                    l1_real_4 = np.real(f4['l1_snr_series'][()] )
                    v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
                    h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                    l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                    v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
                    h1_real_5 = np.real(f5['h1_snr_series'][()] )
                    l1_real_5 = np.real(f5['l1_snr_series'][()] )
                    v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
                    h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
                    l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
                    v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
                
                    h1_real_6 = np.real(f6['h1_snr_series'][()] )
                    l1_real_6 = np.real(f6['l1_snr_series'][()] )
                    v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
                    h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
                    l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
                    v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
            
                    h1_real_7 = np.real(f7['h1_snr_series'][()] )
                    l1_real_7 = np.real(f7['l1_snr_series'][()] )
                    v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
                    h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
                    l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
                    v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
                    h1_real_8 = np.real(f8['h1_snr_series'][()] )
                    l1_real_8 = np.real(f8['l1_snr_series'][()] )
                    v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
                    h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                    l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
                    v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
                    h1_real_9 = np.real(f9['h1_snr_series'][()] )
                    l1_real_9 = np.real(f9['l1_snr_series'][()] )
                    v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
                    h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                    l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
                    v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
                
                    h1_real_10 = np.real(f10['h1_snr_series'][()] )
                    l1_real_10 = np.real(f10['l1_snr_series'][()] )
                    v1_real_10 = np.real(f10['v1_snr_series'][()] )
    
                    h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
                    l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
                    v1_imag_10 = np.imag((f10['v1_snr_series'][()]))
                
                    h1_real_11 = np.real(f11['h1_snr_series'][()] )
                    l1_real_11 = np.real(f11['l1_snr_series'][()] )
                    v1_real_11 = np.real(f11['v1_snr_series'][()] )
    
                    h1_imag_11 = np.imag((f11['h1_snr_series'][()]))
                    l1_imag_11 = np.imag((f11['l1_snr_series'][()]))
                    v1_imag_11 = np.imag((f11['v1_snr_series'][()]))
                
                    h1_real_12 = np.real(f12['h1_snr_series'][()] )
                    l1_real_12 = np.real(f12['l1_snr_series'][()] )
                    v1_real_12 = np.real(f12['v1_snr_series'][()] )
    
                    h1_imag_12 = np.imag((f12['h1_snr_series'][()]))
                    l1_imag_12 = np.imag((f12['l1_snr_series'][()]))
                    v1_imag_12 = np.imag((f12['v1_snr_series'][()]))
                
                    h1_real_13 = np.real(f13['h1_snr_series'][()] )
                    l1_real_13 = np.real(f13['l1_snr_series'][()] )
                    v1_real_13 = np.real(f13['v1_snr_series'][()] )
    
                    h1_imag_13 = np.imag((f13['h1_snr_series'][()]))
                    l1_imag_13 = np.imag((f13['l1_snr_series'][()]))
                    v1_imag_13 = np.imag((f13['v1_snr_series'][()]))
                
            
                    h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9, h1_real_10, h1_real_11, h1_real_12, h1_real_13], axis=0)
                    l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9, l1_real_10, l1_real_11, l1_real_12, l1_real_13], axis=0)
                    v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9, v1_real_10, v1_real_11, v1_real_12, v1_real_13], axis=0)
            
                    h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8,  h1_imag_9, h1_imag_10, h1_imag_11, h1_imag_12, h1_imag_13], axis=0)
                    l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9, l1_imag_10, l1_imag_11, l1_imag_12, l1_imag_13], axis=0)
                    v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9, v1_imag_10, v1_imag_11, v1_imag_12, v1_imag_13], axis=0)
            
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                    f13.close()
                    f14.close()
                    f15.close()
                    f16.close()
                    f17.close()
                    f18.close()
                    f19.close()
                    f20.close()
    
                elif(data_config.train.train_negative_latency_seconds == '15'):
            
                    f1 = h5py.File(data_config.data.BNS.path_train_design_15_sec_1, 'r')
                    f2 = h5py.File(data_config.data.BNS.path_train_design_15_sec_2, 'r')
                    f3 = h5py.File(data_config.data.BNS.path_train_design_15_sec_3, 'r')
                    f4 = h5py.File(data_config.data.BNS.path_train_design_15_sec_4, 'r')
                    f5 = h5py.File(data_config.data.BNS.path_train_design_15_sec_5, 'r')
                    f6 = h5py.File(data_config.data.BNS.path_train_design_15_sec_6, 'r')
                    f7 = h5py.File(data_config.data.BNS.path_train_design_15_sec_7, 'r')
                    f8 = h5py.File(data_config.data.BNS.path_train_design_15_sec_8, 'r')
                    f9 = h5py.File(data_config.data.BNS.path_train_design_15_sec_9, 'r')
                    f10 = h5py.File(data_config.data.BNS.path_train_design_15_sec_10, 'r')
                    f11 = h5py.File(data_config.data.BNS.path_train_design_15_sec_11, 'r')
                    f12 = h5py.File(data_config.data.BNS.path_train_design_15_sec_12, 'r')

                    h1_real_1 = np.real(f1['h1_snr_series'][()] )
                    l1_real_1 = np.real(f1['l1_snr_series'][()] )
                    v1_real_1 = np.real(f1['v1_snr_series'][()] )

                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
                    h1_real_2 = np.real(f2['h1_snr_series'][()] )
                    l1_real_2 = np.real(f2['l1_snr_series'][()] )
                    v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                    h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
                    h1_real_3 = np.real(f3['h1_snr_series'][()] )
                    l1_real_3 = np.real(f3['l1_snr_series'][()] )
                    v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
                    h1_real_4 = np.real(f4['h1_snr_series'][()] )
                    l1_real_4 = np.real(f4['l1_snr_series'][()] )
                    v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
                    h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                    l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                    v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
            
                    h1_real_5 = np.real(f5['h1_snr_series'][0:12000][()] ) # 12000
                    l1_real_5 = np.real(f5['l1_snr_series'][0:12000][()] ) # 12000
                    v1_real_5 = np.real(f5['v1_snr_series'][0:12000][()] ) # 12000
    
                    h1_imag_5 = np.imag((f5['h1_snr_series'][0:12000][()])) # 12000
                    l1_imag_5 = np.imag((f5['l1_snr_series'][0:12000][()])) # 12000
                    v1_imag_5 = np.imag((f5['v1_snr_series'][0:12000][()])) # 12000
                
                    h1_real_6 = np.real(f6['h1_snr_series'][()] )
                    l1_real_6 = np.real(f6['l1_snr_series'][()] )
                    v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
                    h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
                    l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
                    v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
                
                    h1_real_7 = np.real(f7['h1_snr_series'][()] )
                    l1_real_7 = np.real(f7['l1_snr_series'][()] )
                    v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
                    h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
                    l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
                    v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
                    h1_real_8 = np.real(f8['h1_snr_series'][()] )
                    l1_real_8 = np.real(f8['l1_snr_series'][()] )
                    v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
                    h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                    l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
                    v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
                    h1_real_9 = np.real(f9['h1_snr_series'][()] )
                    l1_real_9 = np.real(f9['l1_snr_series'][()] )
                    v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
                    h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                    l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
                    v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
                
                    h1_real_10 = np.real(f10['h1_snr_series'][()] )
                    l1_real_10 = np.real(f10['l1_snr_series'][()] )
                    v1_real_10 = np.real(f10['v1_snr_series'][()] )
    
                    h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
                    l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
                    v1_imag_10 = np.imag((f10['v1_snr_series'][()]))
                
                    h1_real_11 = np.real(f11['h1_snr_series'][()] )
                    l1_real_11 = np.real(f11['l1_snr_series'][()] )
                    v1_real_11 = np.real(f11['v1_snr_series'][()] )
    
                    h1_imag_11 = np.imag((f11['h1_snr_series'][()]))
                    l1_imag_11 = np.imag((f11['l1_snr_series'][()]))
                    v1_imag_11 = np.imag((f11['v1_snr_series'][()]))
                
                    h1_real_12 = np.real(f12['h1_snr_series'][()] )
                    l1_real_12 = np.real(f12['l1_snr_series'][()] )
                    v1_real_12 = np.real(f12['v1_snr_series'][()] )
    
                    h1_imag_12 = np.imag((f12['h1_snr_series'][()]))
                    l1_imag_12 = np.imag((f12['l1_snr_series'][()]))
                    v1_imag_12 = np.imag((f12['v1_snr_series'][()]))
                                
            
                    h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9, h1_real_10, h1_real_11, h1_real_12], axis=0)
                    l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9, l1_real_10, l1_real_11, l1_real_12], axis=0)
                    v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9, v1_real_10, v1_real_11, v1_real_12], axis=0)
            
                    h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9, h1_imag_10, h1_imag_11, h1_imag_12], axis=0)
                    l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9, l1_imag_10, l1_imag_11, l1_imag_12], axis=0)
                    v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9, v1_imag_10, v1_imag_11, v1_imag_12], axis=0)
                    
            
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                        
            
                elif(data_config.train.train_negative_latency_seconds == '30'):
            
                    f1 = h5py.File(data_config.data.BNS.path_train_design_30_sec_1, 'r')
                    f2 = h5py.File(data_config.data.BNS.path_train_design_30_sec_2, 'r')
                    f3 = h5py.File(data_config.data.BNS.path_train_design_30_sec_3, 'r')
                    f4 = h5py.File(data_config.data.BNS.path_train_design_30_sec_4, 'r')
                    f5 = h5py.File(data_config.data.BNS.path_train_design_30_sec_5, 'r')
                    f6 = h5py.File(data_config.data.BNS.path_train_design_30_sec_6 ,'r')
                    f7 = h5py.File(data_config.data.BNS.path_train_design_30_sec_7, 'r')
                    f8 = h5py.File(data_config.data.BNS.path_train_design_30_sec_8, 'r')
                    f9 = h5py.File(data_config.data.BNS.path_train_design_30_sec_9, 'r')
                    f10 = h5py.File(data_config.data.BNS.path_train_design_30_sec_10, 'r')
                    f11 = h5py.File(data_config.data.BNS.path_train_design_30_sec_11, 'r')
                    f12 = h5py.File(data_config.data.BNS.path_train_design_30_sec_12, 'r')

                    h1_real_1 = np.real(f1['h1_snr_series'][()] )
                    l1_real_1 = np.real(f1['l1_snr_series'][()] )
                    v1_real_1 = np.real(f1['v1_snr_series'][()] )

                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
                    h1_real_2 = np.real(f2['h1_snr_series'][()] )
                    l1_real_2 = np.real(f2['l1_snr_series'][()] )
                    v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                    h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
                    h1_real_3 = np.real(f3['h1_snr_series'][()] )
                    l1_real_3 = np.real(f3['l1_snr_series'][()] )
                    v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
                    h1_real_4 = np.real(f4['h1_snr_series'][()] )
                    l1_real_4 = np.real(f4['l1_snr_series'][()] )
                    v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
                    h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                    l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                    v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
            
                    h1_real_5 = np.real(f5['h1_snr_series'][0:26000][()] )
                    l1_real_5 = np.real(f5['l1_snr_series'][0:26000][()] )
                    v1_real_5 = np.real(f5['v1_snr_series'][0:26000][()] )
    
                    h1_imag_5 = np.imag((f5['h1_snr_series'][0:26000][()]))
                    l1_imag_5 = np.imag((f5['l1_snr_series'][0:26000][()]))
                    v1_imag_5 = np.imag((f5['v1_snr_series'][0:26000][()]))
                
                    h1_real_6 = np.real(f6['h1_snr_series'][()] )
                    l1_real_6 = np.real(f6['l1_snr_series'][()] )
                    v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
                    h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
                    l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
                    v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
                
                    h1_real_7 = np.real(f7['h1_snr_series'][()] )
                    l1_real_7 = np.real(f7['l1_snr_series'][()] )
                    v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
                    h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
                    l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
                    v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
                    h1_real_8 = np.real(f8['h1_snr_series'][()] )
                    l1_real_8 = np.real(f8['l1_snr_series'][()] )
                    v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
                    h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                    l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
                    v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
                    h1_real_9 = np.real(f9['h1_snr_series'][()] )
                    l1_real_9 = np.real(f9['l1_snr_series'][()] )
                    v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
                    h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                    l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
                    v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
                
                    h1_real_10 = np.real(f10['h1_snr_series'][()] )
                    l1_real_10 = np.real(f10['l1_snr_series'][()] )
                    v1_real_10 = np.real(f10['v1_snr_series'][()] )
    
                    h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
                    l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
                    v1_imag_10 = np.imag((f10['v1_snr_series'][()]))
                
                    h1_real_11 = np.real(f11['h1_snr_series'][()] )
                    l1_real_11 = np.real(f11['l1_snr_series'][()] )
                    v1_real_11 = np.real(f11['v1_snr_series'][()] )
    
                    h1_imag_11 = np.imag((f11['h1_snr_series'][()]))
                    l1_imag_11 = np.imag((f11['l1_snr_series'][()]))
                    v1_imag_11 = np.imag((f11['v1_snr_series'][()]))
                
                    h1_real_12 = np.real(f12['h1_snr_series'][()] )
                    l1_real_12 = np.real(f12['l1_snr_series'][()] )
                    v1_real_12 = np.real(f12['v1_snr_series'][()] )
    
                    h1_imag_12 = np.imag((f12['h1_snr_series'][()]))
                    l1_imag_12 = np.imag((f12['l1_snr_series'][()]))
                    v1_imag_12 = np.imag((f12['v1_snr_series'][()]))
                
                    h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9, h1_real_10, h1_real_11, h1_real_12], axis=0)
                    l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9, l1_real_10, l1_real_11, l1_real_12], axis=0)
                    v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9, v1_real_10, v1_real_11, v1_real_12], axis=0)
            
                    h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9, h1_imag_10, h1_imag_11, h1_imag_12], axis=0)
                    l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9, l1_imag_10, l1_imag_11, l1_imag_12], axis=0)
                    v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9, v1_imag_10, v1_imag_11, v1_imag_12], axis=0)
            
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()            
                  
                    
                elif(data_config.train.train_negative_latency_seconds == '45'):
            
                    f1 = h5py.File(data_config.data.BNS.path_train_design_45_sec_1, 'r')
                    f2 = h5py.File(data_config.data.BNS.path_train_design_45_sec_2, 'r')
                    f3 = h5py.File(data_config.data.BNS.path_train_design_45_sec_3, 'r')
                    f4 = h5py.File(data_config.data.BNS.path_train_design_45_sec_4, 'r')
                    f5 = h5py.File(data_config.data.BNS.path_train_design_45_sec_5, 'r')
                    f6 = h5py.File(data_config.data.BNS.path_train_design_45_sec_6, 'r')
                    f7 = h5py.File(data_config.data.BNS.path_train_design_45_sec_7, 'r')
                    f8 = h5py.File(data_config.data.BNS.path_train_design_45_sec_8, 'r')
                    f9 = h5py.File(data_config.data.BNS.path_train_design_45_sec_9, 'r')
                    f10 = h5py.File(data_config.data.BNS.path_train_design_45_sec_10, 'r')
                    f11 = h5py.File(data_config.data.BNS.path_train_design_45_sec_11, 'r')
                    f12 = h5py.File(data_config.data.BNS.path_train_design_45_sec_12, 'r')
                    f13 = h5py.File(data_config.data.BNS.path_train_design_45_sec_13, 'r')
                    f14 = h5py.File(data_config.data.BNS.path_train_design_45_sec_14, 'r')
                    f15 = h5py.File(data_config.data.BNS.path_train_design_45_sec_15, 'r')
                    f16 = h5py.File(data_config.data.BNS.path_train_design_45_sec_16, 'r')

                    h1_real_1 = np.real(f1['h1_snr_series'][()] )
                    l1_real_1 = np.real(f1['l1_snr_series'][()] )
                    v1_real_1 = np.real(f1['v1_snr_series'][()] )

                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
                    h1_real_2 = np.real(f2['h1_snr_series'][()] )
                    l1_real_2 = np.real(f2['l1_snr_series'][()] )
                    v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                    h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
                    h1_real_3 = np.real(f3['h1_snr_series'][()] )
                    l1_real_3 = np.real(f3['l1_snr_series'][()] )
                    v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
                    h1_real_4 = np.real(f4['h1_snr_series'][()] )
                    l1_real_4 = np.real(f4['l1_snr_series'][()] )
                    v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
                    h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                    l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                    v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
#                    h1_real_5 = np.real(f5['h1_snr_series'][()] )
#                    l1_real_5 = np.real(f5['l1_snr_series'][()] )
#                    v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
#                    h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
#                    l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
#                    v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
            
                    h1_real_5 = np.real(f5['h1_snr_series'][0:42000][()] )
                    l1_real_5 = np.real(f5['l1_snr_series'][0:42000][()] )
                    v1_real_5 = np.real(f5['v1_snr_series'][0:42000][()] )
    
                    h1_imag_5 = np.imag((f5['h1_snr_series'][0:42000][()]))
                    l1_imag_5 = np.imag((f5['l1_snr_series'][0:42000][()]))
                    v1_imag_5 = np.imag((f5['v1_snr_series'][0:42000][()]))
                
                    h1_real_6 = np.real(f6['h1_snr_series'][()] )
                    l1_real_6 = np.real(f6['l1_snr_series'][()] )
                    v1_real_6 = np.real(f6['v1_snr_series'][()] )

                    h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
                    l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
                    v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
            
                    h1_real_7 = np.real(f7['h1_snr_series'][()] )
                    l1_real_7 = np.real(f7['l1_snr_series'][()] )
                    v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
                    h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
                    l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
                    v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
                    h1_real_8 = np.real(f8['h1_snr_series'][()] )
                    l1_real_8 = np.real(f8['l1_snr_series'][()] )
                    v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
                    h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                    l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
                    v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
                    h1_real_9 = np.real(f9['h1_snr_series'][()] )
                    l1_real_9 = np.real(f9['l1_snr_series'][()] )
                    v1_real_9 = np.real(f9['v1_snr_series'][()] )
                    
                    h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                    l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
                    v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
                    
                    h1_real_10 = np.real(f10['h1_snr_series'][()] )
                    l1_real_10 = np.real(f10['l1_snr_series'][()] )
                    v1_real_10 = np.real(f10['v1_snr_series'][()] )

                    h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
                    l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
                    v1_imag_10 = np.imag((f10['v1_snr_series'][()]))
            
                    h1_real_11 = np.real(f11['h1_snr_series'][()] )
                    l1_real_11 = np.real(f11['l1_snr_series'][()] )
                    v1_real_11 = np.real(f11['v1_snr_series'][()] )
    
                    h1_imag_11 = np.imag((f11['h1_snr_series'][()]))
                    l1_imag_11 = np.imag((f11['l1_snr_series'][()]))
                    v1_imag_11 = np.imag((f11['v1_snr_series'][()]))
                
                    h1_real_12 = np.real(f12['h1_snr_series'][()] )
                    l1_real_12 = np.real(f12['l1_snr_series'][()] )
                    v1_real_12 = np.real(f12['v1_snr_series'][()] )
    
                    h1_imag_12 = np.imag((f12['h1_snr_series'][()]))
                    l1_imag_12 = np.imag((f12['l1_snr_series'][()]))
                    v1_imag_12 = np.imag((f12['v1_snr_series'][()]))
                
                    h1_real_13 = np.real(f13['h1_snr_series'][()] )
                    l1_real_13 = np.real(f13['l1_snr_series'][()] )
                    v1_real_13 = np.real(f13['v1_snr_series'][()] )
   
                    h1_imag_13 = np.imag((f13['h1_snr_series'][()]))
                    l1_imag_13 = np.imag((f13['l1_snr_series'][()]))
                    v1_imag_13 = np.imag((f13['v1_snr_series'][()]))
            
                    h1_real_14 = np.real(f14['h1_snr_series'][()] )
                    l1_real_14 = np.real(f14['l1_snr_series'][()] )
                    v1_real_14 = np.real(f14['v1_snr_series'][()] )
    
                    h1_imag_14 = np.imag((f14['h1_snr_series'][()]))
                    l1_imag_14 = np.imag((f14['l1_snr_series'][()]))
                    v1_imag_14 = np.imag((f14['v1_snr_series'][()]))
                
                    h1_real_15 = np.real(f15['h1_snr_series'][()] )
                    l1_real_15 = np.real(f15['l1_snr_series'][()] )
                    v1_real_15 = np.real(f15['v1_snr_series'][()] )
    
                    h1_imag_15 = np.imag((f15['h1_snr_series'][()]))
                    l1_imag_15 = np.imag((f15['l1_snr_series'][()]))
                    v1_imag_15 = np.imag((f15['v1_snr_series'][()]))
                
                    h1_real_16 = np.real(f16['h1_snr_series'][()] )
                    l1_real_16 = np.real(f16['l1_snr_series'][()] )
                    v1_real_16 = np.real(f16['v1_snr_series'][()] )
    
                    h1_imag_16 = np.imag((f16['h1_snr_series'][()]))
                    l1_imag_16 = np.imag((f16['l1_snr_series'][()]))
                    v1_imag_16 = np.imag((f16['v1_snr_series'][()]))
            
                    h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9, h1_real_10, h1_real_11, h1_real_12, h1_real_13, h1_real_14, h1_real_15, h1_real_16], axis=0)
                    l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9, l1_real_10, l1_real_11, l1_real_12, l1_real_13, l1_real_14, l1_real_15, l1_real_16], axis=0)
                    v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9, v1_real_10, v1_real_11, v1_real_12, v1_real_13, v1_real_14, v1_real_15, v1_real_16], axis=0)
            
                    h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9, h1_imag_10, h1_imag_11, h1_imag_12, h1_imag_13, h1_imag_14, h1_imag_15, h1_imag_16], axis=0)
                    l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9, l1_imag_10, l1_imag_11, l1_imag_12, l1_imag_13, l1_imag_14, l1_imag_15, l1_imag_16], axis=0)
                    v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9, v1_imag_10, v1_imag_11, v1_imag_12, v1_imag_13, v1_imag_14, v1_imag_15, v1_imag_16], axis=0)
            
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                    f13.close()
                    f14.close()
                    f15.close()
                    f16.close()
                    
                elif(data_config.train.train_negative_latency_seconds == '58'):
            
                    f1 = h5py.File(data_config.data.BNS.path_train_design_58_sec_1, 'r')
                    f2 = h5py.File(data_config.data.BNS.path_train_design_58_sec_2, 'r')
                    f3 = h5py.File(data_config.data.BNS.path_train_design_58_sec_3, 'r')
                    f4 = h5py.File(data_config.data.BNS.path_train_design_58_sec_4, 'r')
                    f5 = h5py.File(data_config.data.BNS.path_train_design_58_sec_5, 'r')
                    f6 = h5py.File(data_config.data.BNS.path_train_design_58_sec_6, 'r')
                    f7 = h5py.File(data_config.data.BNS.path_train_design_58_sec_7, 'r')
                    f8 = h5py.File(data_config.data.BNS.path_train_design_58_sec_8, 'r')
                    f9 = h5py.File(data_config.data.BNS.path_train_design_58_sec_9, 'r')

                    h1_real_1 = np.real(f1['h1_snr_series'][()] )
                    l1_real_1 = np.real(f1['l1_snr_series'][()] )
                    v1_real_1 = np.real(f1['v1_snr_series'][()] )

                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
                    h1_real_2 = np.real(f2['h1_snr_series'][()] )
                    l1_real_2 = np.real(f2['l1_snr_series'][()] )
                    v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                    h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
                    h1_real_3 = np.real(f3['h1_snr_series'][()] )
                    l1_real_3 = np.real(f3['l1_snr_series'][()] )
                    v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
                    h1_real_4 = np.real(f4['h1_snr_series'][0:62000][()] )
                    l1_real_4 = np.real(f4['l1_snr_series'][0:62000][()] )
                    v1_real_4 = np.real(f4['v1_snr_series'][0:62000][()] )
    
                    h1_imag_4 = np.imag((f4['h1_snr_series'][0:62000][()]))
                    l1_imag_4 = np.imag((f4['l1_snr_series'][0:62000][()]))
                    v1_imag_4 = np.imag((f4['v1_snr_series'][0:62000][()]))
                
                    h1_real_5 = np.real(f5['h1_snr_series'][()] )
                    l1_real_5 = np.real(f5['l1_snr_series'][()] )
                    v1_real_5 = np.real(f5['v1_snr_series'][()] )

                    h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
                    l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
                    v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
            
                    h1_real_6 = np.real(f6['h1_snr_series'][()] )
                    l1_real_6 = np.real(f6['l1_snr_series'][()] )
                    v1_real_6 = np.real(f6['v1_snr_series'][()] )
    
                    h1_imag_6 = np.imag((f6['h1_snr_series'][()]))
                    l1_imag_6 = np.imag((f6['l1_snr_series'][()]))
                    v1_imag_6 = np.imag((f6['v1_snr_series'][()]))
                
                    h1_real_7 = np.real(f7['h1_snr_series'][()] )
                    l1_real_7 = np.real(f7['l1_snr_series'][()] )
                    v1_real_7 = np.real(f7['v1_snr_series'][()] )
    
                    h1_imag_7 = np.imag((f7['h1_snr_series'][()]))
                    l1_imag_7 = np.imag((f7['l1_snr_series'][()]))
                    v1_imag_7 = np.imag((f7['v1_snr_series'][()]))
                
                    h1_real_8 = np.real(f8['h1_snr_series'][()] )
                    l1_real_8 = np.real(f8['l1_snr_series'][()] )
                    v1_real_8 = np.real(f8['v1_snr_series'][()] )
    
                    h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                    l1_imag_8 = np.imag((f8['l1_snr_series'][()]))
                    v1_imag_8 = np.imag((f8['v1_snr_series'][()]))
                
                    h1_real_9 = np.real(f9['h1_snr_series'][()] )
                    l1_real_9 = np.real(f9['l1_snr_series'][()] )
                    v1_real_9 = np.real(f9['v1_snr_series'][()] )
    
                    h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                    l1_imag_9 = np.imag((f9['l1_snr_series'][()]))
                    v1_imag_9 = np.imag((f9['v1_snr_series'][()]))
            
            
                    h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7, h1_real_8, h1_real_9], axis=0)
                    l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7, l1_real_8, l1_real_9], axis=0)
                    v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5, v1_real_6, v1_real_7, v1_real_8, v1_real_9], axis=0)
            
                    h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7, h1_imag_8, h1_imag_9], axis=0)
                    l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7, l1_imag_8, l1_imag_9], axis=0)
                    v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5, v1_imag_6, v1_imag_7, v1_imag_8, v1_imag_9], axis=0)
            
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                
                
            elif((data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and ((data_config.train.PSD == 'O2'))):
                    
                f1 = h5py.File(data_config.data.BNS.path_train_O2_noise_1, 'r')
                f2 = h5py.File(data_config.data.BNS.path_train_O2_noise_2, 'r')
                f3 = h5py.File(data_config.data.BNS.path_train_O2_noise_3, 'r')
                f4 = h5py.File(data_config.data.BNS.path_train_O2_noise_4, 'r')
                f5 = h5py.File(data_config.data.BNS.path_train_O2_noise_5, 'r')

                h1_real_1 = np.real(f1['h1_snr_series'][()] )
                l1_real_1 = np.real(f1['l1_snr_series'][()] )
                v1_real_1 = np.real(f1['v1_snr_series'][()] )

                h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
            
                h1_real_2 = np.real(f2['h1_snr_series'][()] )
                l1_real_2 = np.real(f2['l1_snr_series'][()] )
                v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                
                h1_real_3 = np.real(f3['h1_snr_series'][()] )
                l1_real_3 = np.real(f3['l1_snr_series'][()] )
                v1_real_3 = np.real(f3['v1_snr_series'][()] )
    
                h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
                
                h1_real_4 = np.real(f4['h1_snr_series'][()] )
                l1_real_4 = np.real(f4['l1_snr_series'][()] )
                v1_real_4 = np.real(f4['v1_snr_series'][()] )
    
                h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                
                h1_real_5 = np.real(f5['h1_snr_series'][()] )
                l1_real_5 = np.real(f5['l1_snr_series'][()] )
                v1_real_5 = np.real(f5['v1_snr_series'][()] )
    
                h1_imag_5 = np.imag((f5['h1_snr_series'][()]))
                l1_imag_5 = np.imag((f5['l1_snr_series'][()]))
                v1_imag_5 = np.imag((f5['v1_snr_series'][()]))
                
                
                h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5], axis=0)
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5], axis=0)
                        
                l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5], axis=0)
                        
                v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5], axis=0)
                v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5], axis=0)
                       

                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
            
    
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O4')):
                
                f1 = h5py.File(data_config.data.BNS.path_train_O4_PSD_1, 'r')
                f2 = h5py.File(data_config.data.BNS.path_train_O4_PSD_2, 'r')
                f3 = h5py.File(data_config.data.BNS.path_train_O4_PSD_3, 'r')

                h1_real_1 = np.real(f1['h1_snr_series'][()] )
                l1_real_1 = np.real(f1['l1_snr_series'][()] )
                v1_real_1 = np.real(f1['v1_snr_series'][()] )

                h1_imag_1 = np.realnp.imag((f1['h1_snr_series'][()]))
                l1_imag_1 = np.realnp.imag((f1['l1_snr_series'][()]))
                v1_imag_1 = np.realnp.imag((f1['v1_snr_series'][()]))
            
                h1_real_2 = np.real(f2['h1_snr_series'][()] )
                l1_real_2 = np.real(f2['l1_snr_series'][()] )
                v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                h1_imag_2 = np.realnp.imag((f2['h1_snr_series'][()]))
                l1_imag_2 = np.realnp.imag((f2['l1_snr_series'][()]))
                v1_imag_2 = np.realnp.imag((f2['v1_snr_series'][()]))
                
                h1_real_3 = np.real(f3['h1_snr_series'][()] )
                l1_real_3 = np.real(f3['l1_snr_series'][()] )
                v1_real_3 = np.real(f3['v1_snr_series'][()] )
        
                h1_imag_3 = np.realnp.imag((f3['h1_snr_series'][()]))
                l1_imag_3 = np.realnp.imag((f3['l1_snr_series'][()]))
                v1_imag_3 = np.realnp.imag((f3['v1_snr_series'][()]))
            
                h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3], axis=0)
                l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3], axis=0)
                v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3], axis=0)
            
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3], axis=0)
                v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3], axis=0)
            
                f1.close()
                f2.close()
                f3.close()
             
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == True) and (data_config.train.PSD == 'O4')):
                
                if(data_config.train.train_negative_latency_seconds == '5'):   
                    
                    f1 = h5py.File(data_config.data.BNS.path_train_O4_PSD_5_sec_1, 'r')
                    f2 = h5py.File(data_config.data.BNS.path_train_O4_PSD_5_sec_2, 'r')
                    f3 = h5py.File(data_config.data.BNS.path_train_O4_PSD_5_sec_3, 'r')
                    f4 = h5py.File(data_config.data.BNS.path_train_O4_PSD_5_sec_4, 'r')
                    f5 = h5py.File(data_config.data.BNS.path_train_O4_PSD_5_sec_5, 'r')

                    h1_real_1 = np.real(f1['h1_snr_series'][()] )
                    l1_real_1 = np.real(f1['l1_snr_series'][()] )
                    v1_real_1 = np.real(f1['v1_snr_series'][()] )

                    h1_imag_1 = np.realnp.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.realnp.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.realnp.imag((f1['v1_snr_series'][()]))
            
                    h1_real_2 = np.real(f2['h1_snr_series'][()] )
                    l1_real_2 = np.real(f2['l1_snr_series'][()] )
                    v1_real_2 = np.real(f2['v1_snr_series'][()] )
    
                    h1_imag_2 = np.realnp.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.realnp.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.realnp.imag((f2['v1_snr_series'][()]))
                
                    h1_real_3 = np.real(f3['h1_snr_series'][()] )
                    l1_real_3 = np.real(f3['l1_snr_series'][()] )
                    v1_real_3 = np.real(f3['v1_snr_series'][()] )
        
                    h1_imag_3 = np.realnp.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.realnp.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.realnp.imag((f3['v1_snr_series'][()]))
                    
                    h1_real_4 = np.real(f4['h1_snr_series'][()] )
                    l1_real_4 = np.real(f4['l1_snr_series'][()] )
                    v1_real_4 = np.real(f4['v1_snr_series'][()] )
        
                    h1_imag_4 = np.realnp.imag((f4['h1_snr_series'][()]))
                    l1_imag_4 = np.realnp.imag((f4['l1_snr_series'][()]))
                    v1_imag_4 = np.realnp.imag((f4['v1_snr_series'][()]))
                    
                    h1_real_5 = np.real(f5['h1_snr_series'][()] )
                    l1_real_5 = np.real(f5['l1_snr_series'][()] )
                    v1_real_5 = np.real(f5['v1_snr_series'][()] )
        
                    h1_imag_5 = np.realnp.imag((f5['h1_snr_series'][()]))
                    l1_imag_5 = np.realnp.imag((f5['l1_snr_series'][()]))
                    v1_imag_5 = np.realnp.imag((f5['v1_snr_series'][()]))
            
                    h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5], axis=0)
                    l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5], axis=0)
                    v1_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3, v1_real_4, v1_real_5], axis=0)
            
                    h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5], axis=0)
                    l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5], axis=0)
                    v1_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3, v1_imag_4, v1_imag_5], axis=0)
            
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                
                
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
    def load_train_2_det_data(data_config):
        """Loads dataset from path"""
        # BNS dataset
        if(data_config.train.train_negative_latency == False):
            if((data_config.train.dataset == 'BNS') and (data_config.train.snr_range_train == 'low')):
                f1 = h5py.File(data_config.data.BNS.path_train_2_det_low_SNR, 'r')
                f2 = h5py.File(data_config.data.BNS.path_train_low_snr_1, 'r')
                f3 = h5py.File(data_config.data.BNS.path_train_low_snr_2, 'r')
                f4 = h5py.File(data_config.data.BNS.path_train_low_snr_3, 'r')  
                f5 = h5py.File(data_config.data.BNS.path_train_1, 'r')
                f6 = h5py.File(data_config.data.BNS.path_train_2, 'r')
                f7 = h5py.File(data_config.data.BNS.path_train_2_det_low_SNR_1, 'r')
                f8 = h5py.File(data_config.data.BNS.path_train_2_det_high_SNR_1, 'r')
                f9 = h5py.File(data_config.data.BNS.path_train_2_det_high_SNR_2, 'r')
            
                h1_real = np.real(f1['h1_snr_series'][()])
                l1_real = np.real(f1['l1_snr_series'][()])
            
                h1_imag = np.imag((f1['h1_snr_series'][()]))
                l1_imag = np.imag((f1['l1_snr_series'][()]))

                h1_real_12k = np.real(f2['h1_snr_series'][0:12000][()])
                l1_real_12k = np.real(f2['l1_snr_series'][0:12000][()])

                h1_imag_12k = np.imag((f2['h1_snr_series'][0:12000][()]))
                l1_imag_12k = np.imag((f2['l1_snr_series'][0:12000][()]))
            
                h1_real_36k = np.real(f3['h1_snr_series'][0:36000][()] )
                l1_real_36k = np.real(f3['l1_snr_series'][0:36000][()] )
    
                h1_imag_36k = np.imag((f3['h1_snr_series'][0:36000][()]))
                l1_imag_36k = np.imag((f3['l1_snr_series'][0:36000][()]))
                
                h1_real_52k = np.real(f4['h1_snr_series'][0:52000][()] )
                l1_real_52k = np.real(f4['l1_snr_series'][0:52000][()] )
              
                h1_imag_52k = np.imag((f4['h1_snr_series'][0:52000][()]))
                l1_imag_52k = np.imag((f4['l1_snr_series'][0:52000][()]))
            
                h1_real_22k = np.real(f5['h1_snr_series'][()])
                l1_real_22k = np.real(f5['l1_snr_series'][()])
    
                h1_imag_22k = np.imag((f5['h1_snr_series'][()]))
                l1_imag_22k = np.imag((f5['l1_snr_series'][()]))
            
                h1_real_86k = np.real(f6['h1_snr_series'][()])
                l1_real_86k = np.real(f6['l1_snr_series'][()])
    
                h1_imag_86k = np.imag((f6['h1_snr_series'][()]))
                l1_imag_86k = np.imag((f6['l1_snr_series'][()]))
            
                h1_real_102k = np.real(f7['h1_snr_series'][()])
                l1_real_102k = np.real(f7['l1_snr_series'][()])
    
                h1_imag_102k = np.imag((f7['h1_snr_series'][()]))
                l1_imag_102k = np.imag((f7['l1_snr_series'][()]))
            
                h1_real_high_1 = np.real(f8['h1_snr_series'][()])
                l1_real_high_1 = np.real(f8['l1_snr_series'][()])
            
                h1_imag_high_1 = np.imag((f8['h1_snr_series'][()]))
                l1_imag_high_1 = np.imag((f8['l1_snr_series'][()]))
            
                h1_real_high_2 = np.real(f9['h1_snr_series'][()])
                l1_real_high_2 = np.real(f9['l1_snr_series'][()])
            
                h1_imag_high_2 = np.imag((f9['h1_snr_series'][()]))
                l1_imag_high_2 = np.imag((f9['l1_snr_series'][()]))
            
                        
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
            elif((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_train == 'low')):
                f1 = h5py.File(data_config.data.NSBH.path_train_1, 'r')
                f2 = h5py.File(data_config.data.NSBH.path_train_2, 'r')
                f3 = h5py.File(data_config.data.NSBH.path_train_3, 'r')
                f4 = h5py.File(data_config.data.NSBH.path_train_4, 'r')
                f5 = h5py.File(data_config.data.NSBH.path_train_low_snr_1, 'r')
                f6 = h5py.File(data_config.data.NSBH.path_train_low_snr_2, 'r')
                f7 = h5py.File(data_config.data.NSBH.path_train_low_snr_3, 'r')
            
                h1_real_52k = np.real(f1['h1_snr_series'][()])
                l1_real_52k = np.real(f1['l1_snr_series'][()])
        
                h1_real_30k = np.real(f2['h1_snr_series'][()])
                l1_real_30k = np.real(f2['l1_snr_series'][()])
            
                h1_real_12k = np.real(f3['h1_snr_series'][()])
                l1_real_12k = np.real(f3['l1_snr_series'][()])
        
                h1_real_6k = np.real(f4['h1_snr_series'][()])
                l1_real_6k = np.real(f4['l1_snr_series'][()])
            
                h1_real_60k = np.real(f5['h1_snr_series'][0:60000][()] )
                l1_real_60k = np.real(f5['l1_snr_series'][0:60000][()] )
            
                h1_real_40k = np.real(f6['h1_snr_series'][0:40000][()] )
                l1_real_40k = np.real(f6['l1_snr_series'][0:40000][()] )
            
                h1_real_72k = np.real(f7['h1_snr_series'][()] )
                l1_real_72k = np.real(f7['l1_snr_series'][()] )
                
                h1_imag_52k = np.imag((f1['h1_snr_series'][()]))
                l1_imag_52k = np.imag((f1['l1_snr_series'][()]))
        
                h1_imag_30k = np.imag((f2['h1_snr_series'][()]))
                l1_imag_30k = np.imag((f2['l1_snr_series'][()]))
        
                h1_imag_12k = np.imag((f3['h1_snr_series'][()]))
                l1_imag_12k = np.imag((f3['l1_snr_series'][()]))
        
                h1_imag_6k = np.imag((f4['h1_snr_series'][()]))
                l1_imag_6k = np.imag((f4['l1_snr_series'][()]))
            
                h1_imag_60k = np.imag((f5['h1_snr_series'][0:60000][()]))
                l1_imag_60k = np.imag((f5['l1_snr_series'][0:60000][()]))
            
                h1_imag_40k = np.imag((f6['h1_snr_series'][0:40000][()]))
                l1_imag_40k = np.imag((f6['l1_snr_series'][0:40000][()]))
            
                h1_imag_72k = np.imag((f7['h1_snr_series'][()]))
                l1_imag_72k = np.imag((f7['l1_snr_series'][()]))
            
            
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
            elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == False) and (data_config.train.snr_range_train == 'low')):
                f1 = h5py.File(data_config.BBH.path_train, 'r')
                f2 = h5py.File(data_config.BBH.path_train_low_SNR, 'r')
            
                h1_real_1 = np.real(f1['h1_snr_series'][2000:200000][()])
                l1_real_1 = np.real(f1['l1_snr_series'][2000:200000][()])
        
                h1_real_2 = np.real(f2['h1_snr_series'][()])
                l1_real_2 = np.real(f2['l1_snr_series'][()])
                
                h1_imag_1 = np.imag((f1['h1_snr_series'][2000:200000][()]))
                l1_imag_1 = np.imag((f1['l1_snr_series'][2000:200000][()]))
        
                h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
            
                h1_real = np.concatenate([h1_real_1, h1_real_2], axis=0)
                l1_real = np.concatenate([l1_real_1, l1_real_2], axis=0)
            
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2], axis=0)
            
                f1.close()
                f2.close()  
            
            elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == True) and (data_config.train.PSD == 'O2')):
            
                f1 = h5py.File(data_config.data.BBH.path_train_real_noise_1, 'r')
                f2 = h5py.File(data_config.data.BBH.path_train_real_noise_2, 'r')
                f3 = h5py.File(data_config.data.BBH.path_train_real_noise_3, 'r')
                f4 = h5py.File(data_config.data.BBH.path_train_real_noise_4, 'r')
                f5 = h5py.File(data_config.data.BBH.path_train_real_noise_5, 'r')
                f6 = h5py.File(data_config.data.BBH.path_train_real_noise_6, 'r')
                f7 = h5py.File(data_config.data.BBH.path_train_real_noise_7, 'r')
                
                f8 = h5py.File(data_config.data.NSBH.path_train_real_noise_1, 'r')
                f9 = h5py.File(data_config.data.NSBH.path_train_real_noise_2, 'r')
                f10 = h5py.File(data_config.data.NSBH.path_train_real_noise_3, 'r')
                f11 = h5py.File(data_config.data.NSBH.path_train_real_noise_4, 'r')
                f12 = h5py.File(data_config.data.NSBH.path_train_real_noise_5, 'r')
               
                
                h1_real_1 = np.real(f1['h1_snr_series'][()] )
                l1_real_1 = np.real(f1['l1_snr_series'][()] )
                
                h1_imag_1 = (np.imag((f1['h1_snr_series'][()])))
                l1_imag_1 = (np.imag((f1['l1_snr_series'][()])))
           
                h1_real_2 = np.real(f2['h1_snr_series'][()] )
                l1_real_2 = np.real(f2['l1_snr_series'][()] )
         
                h1_imag_2 = (np.imag((f2['h1_snr_series'][()])))
                l1_imag_2 = (np.imag((f2['l1_snr_series'][()])))
            
                
                h1_real_3 = np.real(f3['h1_snr_series'][()] )
                l1_real_3 = np.real(f3['l1_snr_series'][()] )
           
                h1_imag_3 = (np.imag((f3['h1_snr_series'][()])))
                l1_imag_3 = (np.imag((f3['l1_snr_series'][()])))
          
                
                h1_real_4 = np.real(f4['h1_snr_series'][()] )
                l1_real_4 = np.real(f4['l1_snr_series'][()] )
            
    
                h1_imag_4 = (np.imag((f4['h1_snr_series'][()])))
                l1_imag_4 = (np.imag((f4['l1_snr_series'][()])))
            
                
                h1_real_5 = np.real(f5['h1_snr_series'][()] )
                l1_real_5 = np.real(f5['l1_snr_series'][()] )
            
                h1_imag_5 = (np.imag((f5['h1_snr_series'][()])))
                l1_imag_5 = (np.imag((f5['l1_snr_series'][()])))
            
            
                h1_real_6 = np.real(f6['h1_snr_series'][()] )
                l1_real_6 = np.real(f6['l1_snr_series'][()] )
            
                h1_imag_6 = (np.imag((f6['h1_snr_series'][()])))
                l1_imag_6 = (np.imag((f6['l1_snr_series'][()])))
            
                
                h1_real_7 = np.real(f7['h1_snr_series'][()] )
                l1_real_7 = np.real(f7['l1_snr_series'][()] )
          
        
                h1_imag_7 = (np.imag((f7['h1_snr_series'][()])))
                l1_imag_7 = (np.imag((f7['l1_snr_series'][()])))
                
                
                h1_real_8 = np.real(f8['h1_snr_series'][()] )
                l1_real_8 = np.real(f8['l1_snr_series'][()] )

                h1_imag_8 = np.imag((f8['h1_snr_series'][()]))
                l1_imag_8 = np.imag((f8['l1_snr_series'][()]))                
            
                h1_real_9 = np.real(f9['h1_snr_series'][()] )
                l1_real_9 = np.real(f9['l1_snr_series'][()] )
    
                h1_imag_9 = np.imag((f9['h1_snr_series'][()]))
                l1_imag_9 = np.imag((f9['l1_snr_series'][()]))                
                
                h1_real_10 = np.real(f10['h1_snr_series'][()] )
                l1_real_10 = np.real(f10['l1_snr_series'][()] )
    
                h1_imag_10 = np.imag((f10['h1_snr_series'][()]))
                l1_imag_10 = np.imag((f10['l1_snr_series'][()]))
                
                h1_real_11 = np.real(f11['h1_snr_series'][()] )
                l1_real_11 = np.real(f11['l1_snr_series'][()] )
               
                h1_imag_11 = np.imag((f11['h1_snr_series'][()]))
                l1_imag_11 = np.imag((f11['l1_snr_series'][()]))                
                
                h1_real_12 = np.real(f12['h1_snr_series'][()] )
                l1_real_12 = np.real(f12['l1_snr_series'][()] )                
    
                h1_imag_12 = np.imag((f12['h1_snr_series'][()]))
                l1_imag_12 = np.imag((f12['l1_snr_series'][()]))
            
            
                h1_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3, h1_real_4, h1_real_5, h1_real_6, h1_real_7], axis=0) 
                l1_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3, l1_real_4, l1_real_5, l1_real_6, l1_real_7], axis=0) 
            
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3, h1_imag_4, h1_imag_5, h1_imag_6, h1_imag_7], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3, l1_imag_4, l1_imag_5, l1_imag_6, l1_imag_7], axis=0)
                       
            
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
                f8.close()
                f9.close()
                f10.close()
                f11.close()
                f12.close()
            
        elif(data_config.train.train_negative_latency == True):
            
            if((data_config.train.dataset == 'BNS') and (data_config.train.train_negative_latency_seconds == '5')):
                
                f1 = h5py.File(data_config.data.BNS.path_train_2_det_negative_latency_5_1, 'r')
                f2 = h5py.File(data_config.data.BNS.path_train_2_det_negative_latency_5_2, 'r')
                
                h1_real_1 = np.real(f1['h1_snr_series'][()])
                l1_real_1 = np.real(f1['l1_snr_series'][()])
        
                h1_real_2 = np.real(f2['h1_snr_series'][()])
                l1_real_2 = np.real(f2['l1_snr_series'][()])
                
                h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
        
                h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
            
                h1_real = np.concatenate([h1_real_1, h1_real_2], axis=0)
                l1_real = np.concatenate([l1_real_1, l1_real_2], axis=0)
            
                h1_imag = np.concatenate([h1_imag_1, h1_imag_2], axis=0)
                l1_imag = np.concatenate([l1_imag_1, l1_imag_2], axis=0)
            
                f1.close()
                f2.close()     
               
            elif((data_config.train.dataset == 'BNS') and (data_config.train.train_negative_latency_seconds == '10')):
                
                f1 = h5py.File(data_config.data.BNS.path_train_2_det_negative_latency_10, 'r')
                
                h1_real = np.real(f1['h1_snr_series'][()])
                l1_real = np.real(f1['l1_snr_series'][()])
                
                h1_imag = np.imag((f1['h1_snr_series'][()]))
                l1_imag = np.imag((f1['l1_snr_series'][()]))
            
                f1.close()                 
        
        
        h1_real = h1_real[:,:,None]
        l1_real = l1_real[:,:,None]
        
        h1_imag = h1_imag[:,:,None]
        l1_imag = l1_imag[:,:,None]
        
        X_train_real = np.concatenate((h1_real, l1_real), axis=2)
        X_train_imag = np.concatenate((h1_imag, l1_imag), axis=2)
        
        return X_train_real, X_train_imag
       
    
    @staticmethod
    def load_train_3_det_parameters(data_config):
        """Loads train parameters from path"""
        #NSBH
        if((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_train == 'high') and (data_config.train.PSD == 'aLIGO')):
            f1 = h5py.File(data_config.parameters.NSBH.path_train_1, 'r')
            f2 = h5py.File(data_config.parameters.NSBH.path_train_2, 'r')
            f3 = h5py.File(data_config.parameters.NSBH.path_train_3, 'r')
            f4 = h5py.File(data_config.parameters.NSBH.path_train_4, 'r')
        
            ra_52k = 2.0*np.pi*f1['ra'][()]
            dec_52k = np.arcsin(1.0-2.0*f1['dec'][()])
                       
            ra_30k = 2.0*np.pi*(f2['ra'][0:30000][()])
            dec_30k = np.arcsin(1.0-2.0*f2['dec'][0:30000][()])
        
            ra_12k = 2.0*np.pi*(f3['ra'][()])
            dec_12k = np.arcsin(1.0-2.0*f3['dec'][()])
            
            ra_6k = 2.0*np.pi*(f4['ra'][()])
            dec_6k = np.arcsin(1.0-2.0*f4['dec'][()])
       
            ra = np.concatenate([ra_52k, ra_30k, ra_12k, ra_6k])
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            dec = np.concatenate([dec_52k, dec_30k, dec_12k, dec_6k])
                    
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            
        elif((data_config.train.dataset == 'NSBH') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'design')):
                
            f1 = h5py.File(data_config.parameters.NSBH.path_train_design_1, 'r')
            f2 = h5py.File(data_config.parameters.NSBH.path_train_design_2, 'r')
            f3 = h5py.File(data_config.parameters.NSBH.path_train_design_3, 'r')
            f4 = h5py.File(data_config.parameters.NSBH.path_train_design_4, 'r')
            f5 = h5py.File(data_config.parameters.NSBH.path_train_design_5, 'r')
                
            f6 = h5py.File(data_config.parameters.NSBH.path_train_1, 'r')
            f7 = h5py.File(data_config.parameters.NSBH.path_train_2, 'r')
            f8 = h5py.File(data_config.parameters.NSBH.path_train_3, 'r')
            f9 = h5py.File(data_config.parameters.NSBH.path_train_4, 'r')
                
            f10 = h5py.File(data_config.parameters.NSBH.path_train_low_snr_1, 'r')
            f11 = h5py.File(data_config.parameters.NSBH.path_train_low_snr_2, 'r')
                
            ra_1 = 2.0*np.pi*f1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
            mass_1_1 = f1['mass1'][()]
            mass_2_1 = f1['mass2'][()]
            spin_1_1 = f1['spin1z'][()]
            spin_2_1 = f1['spin2z'][()]

            inj_snr_1 = f1['Injection_SNR'][()]
                
            ra_2 = 2.0*np.pi*f2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
            mass_1_2 = f2['mass1'][()]
            mass_2_2 = f2['mass2'][()]
            spin_1_2 = f2['spin1z'][()]
            spin_2_2 = f2['spin2z'][()]

            inj_snr_2 = f2['Injection_SNR'][()]
    
            ra_3 = 2.0*np.pi*f3['ra'][()]
            dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
            mass_1_3 = f3['mass1'][()]
            mass_2_3 = f3['mass2'][()]
            spin_1_3 = f3['spin1z'][()]
            spin_2_3 = f3['spin2z'][()]

            inj_snr_3 = f3['Injection_SNR'][()]
    
            ra_4 = 2.0*np.pi*f4['ra'][()]
            dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
            mass_1_4 = f4['mass1'][()]
            mass_2_4 = f4['mass2'][()]
            spin_1_4 = f4['spin1z'][()]
            spin_2_4 = f4['spin2z'][()]

            inj_snr_4 = f4['Injection_SNR'][()]
    
            ra_5 = 2.0*np.pi*f5['ra'][()]
            dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
            mass_1_5 = f5['mass1'][()]
            mass_2_5 = f5['mass2'][()]
            spin_1_5 = f5['spin1z'][()]
            spin_2_5 = f5['spin2z'][()]

            inj_snr_5 = f5['Injection_SNR'][()]
                           
                
            ra_52k = 2.0*np.pi*f6['ra'][()]
            dec_52k = np.arcsin(1.0-2.0*f6['dec'][()])
            mass_1_52k = f6['mass1'][()]
            mass_2_52k = f6['mass2'][()]
            spin_1_52k = f6['spin1z'][()]
            spin_2_52k = f6['spin2z'][()]

            inj_snr_52k = f6['Injection_SNR'][()]
                                               
            
            ra_30k = 2.0*np.pi*(f7['ra'][0:30000][()])
            dec_30k = np.arcsin(1.0-2.0*f7['dec'][0:30000][()])
            mass_1_30k = f7['mass1'][0:30000][()]
            mass_2_30k = f7['mass2'][0:30000][()]
            spin_1_30k = f7['spin1z'][0:30000][()]
            spin_2_30k = f7['spin2z'][0:30000][()]

            inj_snr_30k = f7['Injection_SNR'][0:30000][()]
                
                                
            ra_12k = 2.0*np.pi*(f8['ra'][()])
            dec_12k = np.arcsin(1.0-2.0*f8['dec'][()])
            mass_1_12k = f8['mass1'][()]
            mass_2_12k = f8['mass2'][()]
            spin_1_12k = f8['spin1z'][()]
            spin_2_12k = f8['spin2z'][()]

            inj_snr_12k = f8['Injection_SNR'][()]
                
        
            ra_6k = 2.0*np.pi*(f9['ra'][()])
            dec_6k = np.arcsin(1.0-2.0*f9['dec'][()])
            mass_1_6k = f9['mass1'][()]
            mass_2_6k = f9['mass2'][()]
            spin_1_6k = f9['spin1z'][()]
            spin_2_6k = f9['spin2z'][()]

            inj_snr_6k = f9['Injection_SNR'][()]
                                
                
            ra_60k = 2.0*np.pi*f10['ra'][()]
            dec_60k = np.arcsin(1.0-2.0*f10['dec'][()])
            mass_1_60k = f10['mass1'][()]
            mass_2_60k = f10['mass2'][()]
            spin_1_60k = f10['spin1z'][()]
            spin_2_60k = f10['spin2z'][()]

            inj_snr_60k = f10['Injection_SNR'][()]
                
            
            ra_40k = 2.0*np.pi*(f11['ra'][0:40000][()])
            dec_40k = np.arcsin(1.0-2.0*f11['dec'][0:40000][()])
            mass_1_40k = f11['mass1'][0:40000][()]
            mass_2_40k = f11['mass2'][0:40000][()]
            spin_1_40k = f11['spin1z'][0:40000][()]
            spin_2_40k = f11['spin2z'][0:40000][()]

            inj_snr_40k = f11['Injection_SNR'][0:40000][()]
                
        
            ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_52k, ra_30k, ra_12k, ra_6k, ra_60k, ra_40k], axis=0)
            mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_52k, mass_1_30k, mass_1_12k, mass_1_6k, mass_1_60k, mass_1_40k], axis=0)
            mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_52k, mass_2_30k, mass_2_12k, mass_2_6k, mass_2_60k, mass_2_40k], axis=0)
            spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_52k, spin_1_30k, spin_1_12k, spin_1_6k, spin_1_60k, spin_1_40k], axis=0)
            spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_52k, spin_2_30k, spin_2_12k, spin_2_6k, spin_2_60k, spin_2_40k], axis=0)
            inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_52k, inj_snr_30k, inj_snr_12k, inj_snr_6k, inj_snr_60k, inj_snr_40k], axis=0)
                
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_52k, dec_30k, dec_12k, dec_6k, dec_60k, dec_40k], axis=0)
            
        
        elif((data_config.train.dataset == 'NSBH') and (data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O2')):
                
            f1 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_1, 'r')
            f2 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_2, 'r')
            f3 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_3, 'r')
            f4 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_4, 'r')
            f5 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_5, 'r')
            
            f6 = h5py.File(data_config.parameters.BBH.path_train_real_noise_1, 'r')
            f7 = h5py.File(data_config.parameters.BBH.path_train_real_noise_2, 'r')
            f8 = h5py.File(data_config.parameters.BBH.path_train_real_noise_3, 'r')
            f9 = h5py.File(data_config.parameters.BBH.path_train_real_noise_4, 'r')
            f10 = h5py.File(data_config.parameters.BBH.path_train_real_noise_5, 'r')
                
            ra_1 = 2.0*np.pi*f1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
            mass_1_1 = f1['mass1'][()]
            mass_2_1 = f1['mass2'][()]
            spin_1_1 = f1['spin1z'][()]
            spin_2_1 = f1['spin2z'][()]
            gps_time_1 = f1['gps_time'][()]
               
            ra_2 = 2.0*np.pi*f2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
            mass_1_2 = f2['mass1'][()]
            mass_2_2 = f2['mass2'][()]
            spin_1_2 = f2['spin1z'][()]
            spin_2_2 = f2['spin2z'][()]
            gps_time_2 = f2['gps_time'][()]
    
            ra_3 = 2.0*np.pi*f3['ra'][()]
            dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
            mass_1_3 = f3['mass1'][()]
            mass_2_3 = f3['mass2'][()]
            spin_1_3 = f3['spin1z'][()]
            spin_2_3 = f3['spin2z'][()]
            gps_time_3 = f3['gps_time'][()]
    
            ra_4 = 2.0*np.pi*f4['ra'][()]
            dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
            mass_1_4 = f4['mass1'][()]
            mass_2_4 = f4['mass2'][()]
            spin_1_4 = f4['spin1z'][()]
            spin_2_4 = f4['spin2z'][()]
            gps_time_4 = f4['gps_time'][()]
    
            ra_5 = 2.0*np.pi*f5['ra'][()]
            dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
            mass_1_5 = f5['mass1'][()]
            mass_2_5 = f5['mass2'][()]
            spin_1_5 = f5['spin1z'][()]
            spin_2_5 = f5['spin2z'][()]
            gps_time_5 = f5['gps_time'][()]
                          
                
            ra_6 = 2.0*np.pi*f6['ra'][()]
            dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
            mass_1_6 = f6['mass1'][()]
            mass_2_6 = f6['mass2'][()]
            spin_1_6 = f6['spin1z'][()]
            spin_2_6 = f6['spin2z'][()]
            gps_time_6 = f6['gps_time'][()]
                
            ra_7 = 2.0*np.pi*f7['ra'][()]
            dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
            mass_1_7 = f7['mass1'][()]
            mass_2_7 = f7['mass2'][()]
            spin_1_7 = f7['spin1z'][()]
            spin_2_7 = f7['spin2z'][()]
            gps_time_7 = f7['gps_time'][()]
    
            ra_8 = 2.0*np.pi*f8['ra'][()]
            dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
            mass_1_8 = f8['mass1'][()]
            mass_2_8 = f8['mass2'][()]
            spin_1_8 = f8['spin1z'][()]
            spin_2_8 = f8['spin2z'][()]
            gps_time_8 = f8['gps_time'][()]
    
            ra_9 = 2.0*np.pi*f9['ra'][()]
            dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
            mass_1_9 = f9['mass1'][()]
            mass_2_9 = f9['mass2'][()]
            spin_1_9 = f9['spin1z'][()]
            spin_2_9 = f9['spin2z'][()]
            gps_time_9 = f9['gps_time'][()]
    
            ra_10 = 2.0*np.pi*f10['ra'][()]
            dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
            mass_1_10 = f10['mass1'][()]
            mass_2_10 = f10['mass2'][()]
            spin_1_10 = f10['spin1z'][()]
            spin_2_10 = f10['spin2z'][()]
            gps_time_10 = f10['gps_time'][()]
                          
        
            ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9, ra_10], axis=0)
            dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9, dec_10], axis=0)
            mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_2_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_2_8, mass_1_9, mass_1_10], axis=0)
            mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9, mass_2_10], axis=0)
            spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9, spin_1_10], axis=0)
            spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9, spin_2_10], axis=0)
            gps_time = np.concatenate([gps_time_1, gps_time_2, gps_time_3, gps_time_4, gps_time_5, gps_time_6, gps_time_7, gps_time_8, gps_time_9, gps_time_10], axis=0)
            
                
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
            f7.close()
            f8.close()
            f9.close()
            f10.close()
        
            
        if((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_train == 'low') and (data_config.train.PSD == 'aLIGO')):
            f1 = h5py.File(data_config.parameters.NSBH.path_train_low_snr_1, 'r')
            f2 = h5py.File(data_config.parameters.NSBH.path_train_low_snr_2, 'r')
        
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
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == False) and (data_config.train.PSD == 'aLIGO')):
            f1 = h5py.File(data_config.parameters.BBH.path_train, 'r')

            ra = 2.0*np.pi*f1['ra'][0:100000][()]
            ra = ra - np.pi
            dec = np.arcsin(1.0-2.0*f1['dec'][0:100000][()])

            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
        
            f1.close()
        
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == False) and (data_config.train.PSD == 'aLIGO')):
            f1 = h5py.File(data_config.parameters.BBH.path_train_low_SNR, 'r')

            ra = 2.0*np.pi*f1['ra'][0:100000][()]
            ra = ra - np.pi
            dec = np.arcsin(1.0-2.0*f1['dec'][0:100000][()])

            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
        
            f1.close()
        
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'design')):
                
            f1 = h5py.File(data_config.parameters.BBH.path_train_design_1, 'r')
            f2 = h5py.File(data_config.parameters.BBH.path_train_design_2, 'r')
            f3 = h5py.File(data_config.parameters.BBH.path_train_design_3, 'r')
            f4 = h5py.File(data_config.parameters.BBH.path_train_design_4, 'r')
            f5 = h5py.File(data_config.parameters.BBH.path_train_design_5, 'r')
                
            f6 = h5py.File(data_config.parameters.BBH.path_train, 'r')
            f7 = h5py.File(data_config.parameters.BBH.path_train_low_SNR, 'r')
              
            ra_1 = 2.0*np.pi*f1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
            mass_1_1 = f1['mass1'][()]
            mass_2_1 = f1['mass2'][()]
            spin_1_1 = f1['spin1z'][()]
            spin_2_1 = f1['spin2z'][()]
                
            ra_2 = 2.0*np.pi*f2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
            mass_1_2 = f2['mass1'][()]
            mass_2_2 = f2['mass2'][()]
            spin_1_2 = f2['spin1z'][()]
            spin_2_2 = f2['spin2z'][()]
   
            ra_3 = 2.0*np.pi*f3['ra'][()]
            dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
            mass_1_3 = f3['mass1'][()]
            mass_2_3 = f3['mass2'][()]
            spin_1_3 = f3['spin1z'][()]
            spin_2_3 = f3['spin2z'][()]
   
            ra_4 = 2.0*np.pi*f4['ra'][()]
            dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
            mass_1_4 = f4['mass1'][()]
            mass_2_4 = f4['mass2'][()]
            spin_1_4 = f4['spin1z'][()]
            spin_2_4 = f4['spin2z'][()]
    
            ra_5 = 2.0*np.pi*f5['ra'][()]
            dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
            mass_1_5 = f5['mass1'][()]
            mass_2_5 = f5['mass2'][()]
            spin_1_5 = f5['spin1z'][()]
            spin_2_5 = f5['spin2z'][()]
    
            ra_6 = 2.0*np.pi*f6['ra'][()]
            dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
            mass_1_6 = f6['mass1'][()]
            mass_2_6 = f6['mass2'][()]
            spin_1_6 = f6['spin1z'][()]
            spin_2_6 = f6['spin2z'][()]
   
            ra_7 = 2.0*np.pi*f7['ra'][()]
            dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
            mass_1_7 = f7['mass1'][()]
            mass_2_7 = f7['mass2'][()]
            spin_1_7 = f7['spin1z'][()]
            spin_2_7 = f7['spin2z'][()]                          
        
#            ra = np.concatenate([ra_1, ra_2, ra_3, ra_6, ra_7], axis=0)
#            dec = np.concatenate([dec_1, dec_2, dec_3, dec_6, dec_7], axis=0)
#            mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_6, mass_1_7], axis=0)
#            mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_6, mass_2_7], axis=0)
#            spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_6, spin_1_7], axis=0)
#            spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_6, spin_2_7], axis=0)
            
            ra = ra_6
            dec = dec_6
            mass_1 = mass_1_6
            mass_2 = mass_2_6
            spin_1 = spin_1_6
            spin_2 = spin_2_6
        
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            
                
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O2')):
                
            f1 = h5py.File(data_config.parameters.BBH.path_train_real_noise_1, 'r')
            f2 = h5py.File(data_config.parameters.BBH.path_train_real_noise_2, 'r')
            f3 = h5py.File(data_config.parameters.BBH.path_train_real_noise_3, 'r')
            f4 = h5py.File(data_config.parameters.BBH.path_train_real_noise_4, 'r')
            f5 = h5py.File(data_config.parameters.BBH.path_train_real_noise_5, 'r')
            f6 = h5py.File(data_config.parameters.BBH.path_train_real_noise_6, 'r')
            f7 = h5py.File(data_config.parameters.BBH.path_train_real_noise_7, 'r')
            f8 = h5py.File(data_config.parameters.BBH.path_train_real_noise_8, 'r')
            f9 = h5py.File(data_config.parameters.BBH.path_train_real_noise_9, 'r')
                
            ra_1 = 2.0*np.pi*f1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
            mass_1_1 = f1['mass1'][()]
            mass_2_1 = f1['mass2'][()]
            spin_1_1 = f1['spin1z'][()]
            spin_2_1 = f1['spin2z'][()]
            gps_time_1 = f1['gps_time'][()]
                
            ra_2 = 2.0*np.pi*f2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
            mass_1_2 = f2['mass1'][()]
            mass_2_2 = f2['mass2'][()]
            spin_1_2 = f2['spin1z'][()]
            spin_2_2 = f2['spin2z'][()]
            gps_time_2 = f2['gps_time'][()]
    
            ra_3 = 2.0*np.pi*f3['ra'][()]
            dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
            mass_1_3 = f3['mass1'][()]
            mass_2_3 = f3['mass2'][()]
            spin_1_3 = f3['spin1z'][()]
            spin_2_3 = f3['spin2z'][()]
            gps_time_3 = f3['gps_time'][()]
  
            ra_4 = 2.0*np.pi*f4['ra'][()]
            dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
            mass_1_4 = f4['mass1'][()]
            mass_2_4 = f4['mass2'][()]
            spin_1_4 = f4['spin1z'][()]
            spin_2_4 = f4['spin2z'][()]
            gps_time_4 = f4['gps_time'][()]
    
            ra_5 = 2.0*np.pi*f5['ra'][()]
            dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
            mass_1_5 = f5['mass1'][()]
            mass_2_5 = f5['mass2'][()]
            spin_1_5 = f5['spin1z'][()]
            spin_2_5 = f5['spin2z'][()]
            gps_time_5 = f5['gps_time'][()]

            ra_6 = 2.0*np.pi*f6['ra'][()]
            dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
            mass_1_6 = f6['mass1'][()]
            mass_2_6 = f6['mass2'][()]
            spin_1_6 = f6['spin1z'][()]
            spin_2_6 = f6['spin2z'][()]
            gps_time_6 = f6['gps_time'][()]


            ra_7 = 2.0*np.pi*f7['ra'][()]
            dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
            mass_1_7 = f7['mass1'][()]
            mass_2_7 = f7['mass2'][()]
            spin_1_7 = f7['spin1z'][()]
            spin_2_7 = f7['spin2z'][()]
            gps_time_7 = f7['gps_time'][()]

            ra_8 = 2.0*np.pi*f8['ra'][()]
            dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
            mass_1_8 = f8['mass1'][()]
            mass_2_8 = f8['mass2'][()]
            spin_1_8 = f8['spin1z'][()]
            spin_2_8 = f8['spin2z'][()]
            gps_time_8 = f8['gps_time'][()]

            ra_9 = 2.0*np.pi*f9['ra'][()]
            dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
            mass_1_9 = f9['mass1'][()]
            mass_2_9 = f9['mass2'][()]
            spin_1_9 = f9['spin1z'][()]
            spin_2_9 = f9['spin2z'][()]
            gps_time_9 = f9['gps_time'][()]
                          
        
            ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9], axis=0)
            dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9], axis=0)
            mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_2_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9], axis=0)
            mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9], axis=0)
            spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9], axis=0)
            spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9], axis=0)
            gps_time = np.concatenate([gps_time_1, gps_time_2, gps_time_3, gps_time_4, gps_time_5, gps_time_6, gps_time_7, gps_time_8, gps_time_9], axis=0)
            
        
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
            f7.close()
            f8.close()
            f9.close()
        
        
        elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O3')):
                
            f1 = h5py.File(data_config.parameters.BBH.path_train_O3_noise_1, 'r')
            f2 = h5py.File(data_config.parameters.BBH.path_train_O3_noise_2, 'r')
            f3 = h5py.File(data_config.parameters.BBH.path_train_O3_noise_3, 'r')
            f4 = h5py.File(data_config.parameters.BBH.path_train_O3_noise_4, 'r')
            

            ra_1 = 2.0*np.pi*f1['ra'][()]
            dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
            mass_1_1 = f1['mass1'][()]
            mass_2_1 = f1['mass2'][()]
            spin_1_1 = f1['spin1z'][()]
            spin_2_1 = f1['spin2z'][()]
            gps_time_1 = f1['gps_time'][()]
               
            ra_2 = 2.0*np.pi*f2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
            mass_1_2 = f2['mass1'][()]
            mass_2_2 = f2['mass2'][()]
            spin_1_2 = f2['spin1z'][()]
            spin_2_2 = f2['spin2z'][()]
            gps_time_2 = f2['gps_time'][()]


            ra_3 = 2.0*np.pi*f3['ra'][()]
            dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
            mass_1_3 = f3['mass1'][()]
            mass_2_3 = f3['mass2'][()]
            spin_1_3 = f3['spin1z'][()]
            spin_2_3 = f3['spin2z'][()]
            gps_time_3 = f3['gps_time'][()]
                
            ra_4 = 2.0*np.pi*f4['ra'][()]
            dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
            mass_1_4 = f4['mass1'][()]
            mass_2_4 = f4['mass2'][()]
            spin_1_4 = f4['spin1z'][()]
            spin_2_4 = f4['spin2z'][()]
            gps_time_4 = f4['gps_time'][()]

            ra = np.concatenate([ra_1, ra_2, ra_3, ra_4], axis=0)
            dec = np.concatenate([dec_1, dec_2, dec_3, dec_4], axis=0)
            mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4], axis=0)
            mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4], axis=0)
            spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4], axis=0)
            spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4], axis=0)
            gps_time = np.concatenate([gps_time_1, gps_time_2, gps_time_3, gps_time_4], axis=0)
            
        
            ra = ra - np.pi
            ra_x = np.cos(ra)
            ra_y = np.sin(ra)
            
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            
        
        #BNS
        elif(data_config.train.dataset == 'BNS'):
            if((data_config.train.snr_range_train == 'high') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and ((data_config.train.PSD == 'aLIGO') or (data_config.train.PSD == 'design'))):
                f1 = h5py.File(data_config.parameters.BNS.path_train_1, 'r')
                f2 = h5py.File(data_config.parameters.BNS.path_train_2, 'r')
            
                ra_22k = 2.0*np.pi*f1['ra'][0:22000][()]
                dec_22k = np.arcsin(1.0-2.0*f1['dec'][0:22000][()])
                mass_1_22k = f1['mass1'][0:22000][()]
                mass_2_22k = f1['mass2'][0:22000][()]
                spin_1_22k = f1['spin1z'][0:22000][()]
                spin_2_22k = f1['spin2z'][0:22000][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_22k = f1['Injection_SNR'][0:22000][()]

  
                ra_86k = 2.0*np.pi*f2['ra'][0:86000][()]
                dec_86k = np.arcsin(1.0-2.0*f2['dec'][0:86000][()])
                mass_1_86k = f2['mass1'][0:86000][()]
                mass_2_86k = f2['mass2'][0:86000][()]
                spin_1_86k = f2['spin1z'][0:86000][()]
                spin_2_86k = f2['spin2z'][0:86000][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_86k = f2['Injection_SNR'][0:86000][()]

           
                ra = np.concatenate([ra_22k, ra_86k], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_22k, dec_86k], axis=0)
               
                mass_1 = np.concatenate([mass_1_22k, mass_1_86k], axis=0)
                mass_2 = np.concatenate([mass_2_22k, mass_2_86k], axis=0)
                spin_1 = np.concatenate([spin_1_22k, spin_2_86k], axis=0)
                spin_2 = np.concatenate([spin_2_22k, spin_2_86k], axis=0)
                inj_snr = np.concatenate([inj_snr_22k, inj_snr_86k], axis=0)
                
                f1.close()
                f2.close()
            
                
            elif((data_config.train.snr_range_train == 'low') and (data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and ((data_config.train.PSD == 'aLIGO'))):
                f1 = h5py.File(data_config.parameters.BNS.path_train_low_snr_1, 'r')
                f2 = h5py.File(data_config.parameters.BNS.path_train_low_snr_2, 'r')
                f3 = h5py.File(data_config.parameters.BNS.path_train_low_snr_3, 'r')
            
                ra_12k = 2.0*np.pi*f1['ra'][0:12000][()]
                dec_12k = np.arcsin(1.0-2.0*f1['dec'][0:12000][()])
               
                mass_1_12k = f1['mass1'][0:12000][()]
                mass_2_12k = f1['mass2'][0:12000][()]
                spin_1_12k = f1['spin1z'][0:12000][()]
                spin_2_12k = f1['spin2z'][0:12000][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_12k = f1['Injection_SNR'][0:12000][()]
        
                ra_36k = 2.0*np.pi*f2['ra'][0:36000][()]
                dec_36k = np.arcsin(1.0-2.0*f2['dec'][0:36000][()])
               
                mass_1_36k = f2['mass1'][0:36000][()]
                mass_2_36k = f2['mass2'][0:36000][()]
                spin_1_36k = f2['spin1z'][0:36000][()]
                spin_2_36k = f2['spin2z'][0:36000][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_36k = f2['Injection_SNR'][0:36000][()]
                
                ra_52k = 2.0*np.pi*f3['ra'][0:52000][()]
                dec_52k = np.arcsin(1.0-2.0*f3['dec'][0:52000][()])
               
                mass_1_52k = f3['mass1'][0:52000][()]
                mass_2_52k = f3['mass2'][0:52000][()]
                spin_1_52k = f3['spin1z'][0:52000][()]
                spin_2_52k = f3['spin2z'][0:52000][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_52k = f3['Injection_SNR'][0:52000][()]
            
                ra = np.concatenate([ra_12k, ra_36k, ra_52k], axis=0)
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
                dec = np.concatenate([dec_12k, dec_36k, dec_52k], axis=0)
               
                mass_1 = np.concatenate([mass_1_12k, mass_1_36k, mass_1_52k], axis=0)
                mass_2 = np.concatenate([mass_2_12k, mass_2_36k, mass_2_52k], axis=0)
                spin_1 = np.concatenate([spin_1_12k, spin_1_36k, spin_1_52k], axis=0)
                spin_2 = np.concatenate([spin_2_12k, spin_2_36k, spin_2_52k], axis=0)
#               inc = np.concatenate([inc_1, inc_2, inc_3, inc_4, inc_5, inc_6, inc_7, inc_8, inc_9, inc_10, inc_11], axis=0)
                inj_snr = np.concatenate([inj_snr_12k, inj_snr_36k, inj_snr_52k], axis=0)
                
        
                f1.close()
                f2.close()
                f3.close()
                
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'design')):
                    
                f1 = h5py.File(data_config.parameters.BNS.path_train_design_1, 'r')
                f2 = h5py.File(data_config.parameters.BNS.path_train_design_2, 'r')
                f3 = h5py.File(data_config.parameters.BNS.path_train_design_3, 'r')
                f4 = h5py.File(data_config.parameters.BNS.path_train_design_4, 'r')
                f5 = h5py.File(data_config.parameters.BNS.path_train_design_5, 'r')
                f6 = h5py.File(data_config.parameters.BNS.path_train_design_6, 'r')
                f7 = h5py.File(data_config.parameters.BNS.path_train_design_7, 'r')
                f8 = h5py.File(data_config.parameters.BNS.path_train_design_8, 'r')
                f9 = h5py.File(data_config.parameters.BNS.path_train_design_9, 'r')
                f10 = h5py.File(data_config.parameters.BNS.path_train_design_10, 'r')
                    
                f11 = h5py.File(data_config.parameters.BNS.path_train_1, 'r')
                f12 = h5py.File(data_config.parameters.BNS.path_train_2, 'r')
                    
                f13 = h5py.File(data_config.parameters.BNS.path_train_design_high_SNR_1, 'r')
                f14 = h5py.File(data_config.parameters.BNS.path_train_design_high_SNR_2, 'r')
            
                ra_11 = 2.0*np.pi*f11['ra'][()]
                dec_11 = np.arcsin(1.0-2.0*f11['dec'][()])
                mass_1_11 = f11['mass1'][()]
                mass_2_11 = f11['mass2'][()]
                spin_1_11 = f11['spin1z'][()]
                spin_2_11 = f11['spin2z'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_11 = f11['Injection_SNR'][()]

        #            ra_22k = f1['ra'][0:22000][()]
        #            dec_22k = f1['dec'][0:22000][()]
        
                ra_12 = 2.0*np.pi*f12['ra'][()]
                dec_12 = np.arcsin(1.0-2.0*f12['dec'][()])
                mass_1_12 = f12['mass1'][()]
                mass_2_12 = f12['mass2'][()]
                spin_1_12 = f12['spin1z'][()]
                spin_2_12 = f12['spin2z'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_12 = f12['Injection_SNR'][()]
            
                ra_1 = 2.0*np.pi*f1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                mass_1_1 = f1['mass1'][()]
                mass_2_1 = f1['mass2'][()]
                spin_1_1 = f1['spin1z'][()]
                spin_2_1 = f1['spin2z'][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_1 = f1['Injection_SNR'][()]
        
                ra_2 = 2.0*np.pi*f2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                mass_1_2 = f2['mass1'][()]
                mass_2_2 = f2['mass2'][()]
                spin_1_2 = f2['spin1z'][()]
                spin_2_2 = f2['spin2z'][()]
#                    inc_2 = f2['inclination'][0:50000][()]
                inj_snr_2 = f2['Injection_SNR'][()]
                
                ra_3 = 2.0*np.pi*f3['ra'][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                mass_1_3 = f3['mass1'][()]
                mass_2_3 = f3['mass2'][()]
                spin_1_3 = f3['spin1z'][()]
                spin_2_3 = f3['spin2z'][()]
#                    inc_3 = f3['inclination'][0:50000][()]
                inj_snr_3 = f3['Injection_SNR'][()]
                    
                ra_4 = 2.0*np.pi*f4['ra'][()]
                dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                mass_1_4 = f4['mass1'][()]
                mass_2_4 = f4['mass2'][()]
                spin_1_4 = f4['spin1z'][()]
                spin_2_4 = f4['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                inj_snr_4 = f4['Injection_SNR'][()]
                    
                ra_5 = 2.0*np.pi*f5['ra'][()]
                dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                mass_1_5 = f5['mass1'][()]
                mass_2_5 = f5['mass2'][()]
                spin_1_5 = f5['spin1z'][()]
                spin_2_5 = f5['spin2z'][()]
#                    inc_5 = f5['inclination'][()]
                inj_snr_5 = f5['Injection_SNR'][()]
                    
                ra_6 = 2.0*np.pi*f6['ra'][()]
                dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                mass_1_6 = f6['mass1'][()]
                mass_2_6 = f6['mass2'][()]
                spin_1_6 = f6['spin1z'][()]
                spin_2_6 = f6['spin2z'][()]
#                    inc_6 = f6['inclination'][()]
                inj_snr_6 = f6['Injection_SNR'][()]
                    
                ra_7 = 2.0*np.pi*f7['ra'][()]
                dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                mass_1_7 = f7['mass1'][()]
                mass_2_7 = f7['mass2'][()]
                spin_1_7 = f7['spin1z'][()]
                spin_2_7 = f7['spin2z'][()]
#                    inc_7 = f7['inclination'][()]
                inj_snr_7 = f7['Injection_SNR'][()]
                    
                ra_8 = 2.0*np.pi*f8['ra'][()]
                dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                mass_1_8 = f8['mass1'][()]
                mass_2_8 = f8['mass2'][()]
                spin_1_8 = f8['spin1z'][()]
                spin_2_8 = f8['spin2z'][()]
#                    inc_8 = f8['inclination'][()]
                inj_snr_8 = f8['Injection_SNR'][()]
                    
                ra_9 = 2.0*np.pi*f9['ra'][()]
                dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                mass_1_9 = f9['mass1'][()]
                mass_2_9 = f9['mass2'][()]
                spin_1_9 = f9['spin1z'][()]
                spin_2_9 = f9['spin2z'][()]
#                    inc_9 = f9['inclination'][()]
                inj_snr_9 = f9['Injection_SNR'][()]
                    
                ra_10 = 2.0*np.pi*f10['ra'][()]
                dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
                mass_1_10 = f10['mass1'][()]
                mass_2_10 = f10['mass2'][()]
                spin_1_10 = f10['spin1z'][()]
                spin_2_10 = f10['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                inj_snr_10 = f10['Injection_SNR'][()]
    
                ra_13 = 2.0*np.pi*f13['ra'][()]
                dec_13 = np.arcsin(1.0-2.0*f13['dec'][()])
                mass_1_13 = f13['mass1'][()]
                mass_2_13 = f13['mass2'][()]
                spin_1_13 = f13['spin1z'][()]
                spin_2_13 = f13['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                inj_snr_13 = f13['Injection_SNR'][()]
    
                ra_14 = 2.0*np.pi*f14['ra'][()]
                dec_14 = np.arcsin(1.0-2.0*f14['dec'][()])
                mass_1_14 = f14['mass1'][()]
                mass_2_14 = f14['mass2'][()]
                spin_1_14 = f14['spin1z'][()]
                spin_2_14 = f14['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                inj_snr_14 = f14['Injection_SNR'][()]
    
    
                ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9, ra_10, ra_11, ra_12], axis=0)
                dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9, dec_10, dec_11, dec_12], axis=0)
                mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9, mass_1_10, mass_1_11, mass_1_12], axis=0)
                mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9, mass_2_10, mass_2_11, mass_2_12], axis=0)
                spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9, spin_1_10, spin_1_11, spin_1_12], axis=0)
                spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9, spin_2_10, spin_2_11, spin_2_12], axis=0)
                inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_6, inj_snr_7, inj_snr_8, inj_snr_9, inj_snr_10, inj_snr_11, inj_snr_12], axis=0)
    
    

                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)

                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                f6.close()
                f7.close()
                f8.close()
                f9.close()
                f10.close()
                f11.close()
                f12.close()
                f13.close()
                f14.close()

    
                
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == True) and (data_config.train.PSD == 'aLIGO')):
                
                if(data_config.train.train_negative_latency_seconds == '5'): 
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_5_sec)
              
                    ra = 2.0*np.pi*f1['ra'][()]
                    dec = np.arcsin(1.0-2.0*f1['dec'][()])
            
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
        
                    f1.close()
                 
                elif(data_config.train.train_negative_latency_seconds == '10'):
                 
                    f1 = h5py.File(data_config.parameters.BNS.path_train_10_sec)
              
                    ra = 2.0*np.pi*f1['ra'][()]
                    dec = np.arcsin(1.0-2.0*f1['dec'][()])
            
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
        
                    f1.close()
            
            
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == False) and (data_config.train.PSD == 'O4')):
                    
                f1 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_1, 'r')
                f2 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_2, 'r')
                f3 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_3, 'r')
              
                ra_1 = 2.0*np.pi*f1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                
                ra_2 = 2.0*np.pi*f2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                
                ra_3 = 2.0*np.pi*f3['ra'][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                
                ra = np.concatenate([ra_1, ra_2, ra_3], axis=0)
                
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
                
                dec = np.concatenate([dec_1, dec_2, dec_3], axis=0)
        
                f1.close()
                f2.close()
                f3.close()
                
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == True) and (data_config.train.PSD == 'O4')):
                
                if(data_config.train.train_negative_latency_seconds == '5'):
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_5_sec_1, 'r')
                    f2 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_5_sec_2, 'r')
                    f3 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_5_sec_3, 'r')
                    f4 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_5_sec_4, 'r')
                    f5 = h5py.File(data_config.parameters.BNS.path_train_O4_PSD_5_sec_5, 'r')
              
                    ra_1 = 2.0*np.pi*f1['ra'][()]
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                
                    ra_2 = 2.0*np.pi*f2['ra'][()]
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                
                    ra_3 = 2.0*np.pi*f3['ra'][()]
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                    
                    ra_4 = 2.0*np.pi*f4['ra'][()]
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                    
                    ra_5 = 2.0*np.pi*f5['ra'][()]
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                
                    ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5], axis=0)
                
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                
                    dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5], axis=0)
        
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    
              
            elif((data_config.train.train_real == False) and (data_config.train.train_negative_latency == True) and ((data_config.train.PSD == 'design'))):
                
                if(data_config.train.train_negative_latency_seconds == '0'):
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_design_1, 'r')
                    f2 = h5py.File(data_config.parameters.BNS.path_train_design_2, 'r')
                    f3 = h5py.File(data_config.parameters.BNS.path_train_design_3, 'r')
                    f4 = h5py.File(data_config.parameters.BNS.path_train_design_4, 'r')
                    f5 = h5py.File(data_config.parameters.BNS.path_train_design_5, 'r')
                    f6 = h5py.File(data_config.parameters.BNS.path_train_design_6, 'r')
                    f7 = h5py.File(data_config.parameters.BNS.path_train_design_7, 'r')
                    f8 = h5py.File(data_config.parameters.BNS.path_train_design_8, 'r')
                    f9 = h5py.File(data_config.parameters.BNS.path_train_design_9, 'r')
                    f10 = h5py.File(data_config.parameters.BNS.path_train_design_10, 'r')
                    
                    f11 = h5py.File(data_config.parameters.BNS.path_train_1, 'r')
                    f12 = h5py.File(data_config.parameters.BNS.path_train_2, 'r')
                    
                    f13 = h5py.File(data_config.parameters.BNS.path_train_design_high_SNR_1, 'r')
                    f14 = h5py.File(data_config.parameters.BNS.path_train_design_high_SNR_2, 'r')
            
                    ra_11 = 2.0*np.pi*f11['ra'][()]
                    dec_11 = np.arcsin(1.0-2.0*f11['dec'][()])
                    mass_1_11 = f11['mass1'][()]
                    mass_2_11 = f11['mass2'][()]
                    spin_1_11 = f11['spin1z'][()]
                    spin_2_11 = f11['spin2z'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
                    inj_snr_11 = f11['Injection_SNR'][()]

        #            ra_22k = f1['ra'][0:22000][()]
        #            dec_22k = f1['dec'][0:22000][()]
        
                    ra_12 = 2.0*np.pi*f12['ra'][()]
                    dec_12 = np.arcsin(1.0-2.0*f12['dec'][()])
                    mass_1_12 = f12['mass1'][()]
                    mass_2_12 = f12['mass2'][()]
                    spin_1_12 = f12['spin1z'][()]
                    spin_2_12 = f12['spin2z'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
                    inj_snr_12 = f12['Injection_SNR'][()]
            
                    ra_1 = 2.0*np.pi*f1['ra'][()]
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                    mass_1_1 = f1['mass1'][()]
                    mass_2_1 = f1['mass2'][()]
                    spin_1_1 = f1['spin1z'][()]
                    spin_2_1 = f1['spin2z'][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                    inj_snr_1 = f1['Injection_SNR'][()]
        
                    ra_2 = 2.0*np.pi*f2['ra'][()]
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                    mass_1_2 = f2['mass1'][()]
                    mass_2_2 = f2['mass2'][()]
                    spin_1_2 = f2['spin1z'][()]
                    spin_2_2 = f2['spin2z'][()]
#                    inc_2 = f2['inclination'][0:50000][()]
                    inj_snr_2 = f2['Injection_SNR'][()]
                
                    ra_3 = 2.0*np.pi*f3['ra'][()]
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                    mass_1_3 = f3['mass1'][()]
                    mass_2_3 = f3['mass2'][()]
                    spin_1_3 = f3['spin1z'][()]
                    spin_2_3 = f3['spin2z'][()]
#                    inc_3 = f3['inclination'][0:50000][()]
                    inj_snr_3 = f3['Injection_SNR'][()]
                    
                    ra_4 = 2.0*np.pi*f4['ra'][()]
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                    mass_1_4 = f4['mass1'][()]
                    mass_2_4 = f4['mass2'][()]
                    spin_1_4 = f4['spin1z'][()]
                    spin_2_4 = f4['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_4 = f4['Injection_SNR'][()]
                    
                    ra_5 = 2.0*np.pi*f5['ra'][()]
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                    mass_1_5 = f5['mass1'][()]
                    mass_2_5 = f5['mass2'][()]
                    spin_1_5 = f5['spin1z'][()]
                    spin_2_5 = f5['spin2z'][()]
#                    inc_5 = f5['inclination'][()]
                    inj_snr_5 = f5['Injection_SNR'][()]
                    
                    ra_6 = 2.0*np.pi*f6['ra'][()]
                    dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                    mass_1_6 = f6['mass1'][()]
                    mass_2_6 = f6['mass2'][()]
                    spin_1_6 = f6['spin1z'][()]
                    spin_2_6 = f6['spin2z'][()]
#                    inc_6 = f6['inclination'][()]
                    inj_snr_6 = f6['Injection_SNR'][()]
                    
                    ra_7 = 2.0*np.pi*f7['ra'][()]
                    dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                    mass_1_7 = f7['mass1'][()]
                    mass_2_7 = f7['mass2'][()]
                    spin_1_7 = f7['spin1z'][()]
                    spin_2_7 = f7['spin2z'][()]
#                    inc_7 = f7['inclination'][()]
                    inj_snr_7 = f7['Injection_SNR'][()]
                    
                    ra_8 = 2.0*np.pi*f8['ra'][()]
                    dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                    mass_1_8 = f8['mass1'][()]
                    mass_2_8 = f8['mass2'][()]
                    spin_1_8 = f8['spin1z'][()]
                    spin_2_8 = f8['spin2z'][()]
#                    inc_8 = f8['inclination'][()]
                    inj_snr_8 = f8['Injection_SNR'][()]
                    
                    ra_9 = 2.0*np.pi*f9['ra'][()]
                    dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                    mass_1_9 = f9['mass1'][()]
                    mass_2_9 = f9['mass2'][()]
                    spin_1_9 = f9['spin1z'][()]
                    spin_2_9 = f9['spin2z'][()]
#                    inc_9 = f9['inclination'][()]
                    inj_snr_9 = f9['Injection_SNR'][()]
                    
                    ra_10 = 2.0*np.pi*f10['ra'][()]
                    dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
                    mass_1_10 = f10['mass1'][()]
                    mass_2_10 = f10['mass2'][()]
                    spin_1_10 = f10['spin1z'][()]
                    spin_2_10 = f10['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                    inj_snr_10 = f10['Injection_SNR'][()]
    
                    ra_13 = 2.0*np.pi*f13['ra'][()]
                    dec_13 = np.arcsin(1.0-2.0*f13['dec'][()])
                    mass_1_13 = f13['mass1'][()]
                    mass_2_13 = f13['mass2'][()]
                    spin_1_13 = f13['spin1z'][()]
                    spin_2_13 = f13['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                    inj_snr_13 = f13['Injection_SNR'][()]
    
                    ra_14 = 2.0*np.pi*f14['ra'][()]
                    dec_14 = np.arcsin(1.0-2.0*f14['dec'][()])
                    mass_1_14 = f14['mass1'][()]
                    mass_2_14 = f14['mass2'][()]
                    spin_1_14 = f14['spin1z'][()]
                    spin_2_14 = f14['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                    inj_snr_14 = f14['Injection_SNR'][()]
    
    
                    ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9, ra_10, ra_11, ra_12], axis=0)
                    dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9, dec_10, dec_11, dec_12], axis=0)
                    mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9, mass_1_10, mass_1_11, mass_1_12], axis=0)
                    mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9, mass_2_10, mass_2_11, mass_2_12], axis=0)
                    spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9, spin_1_10, spin_1_11, spin_1_12], axis=0)
                    spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9, spin_2_10, spin_2_11, spin_2_12], axis=0)
                    inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_6, inj_snr_7, inj_snr_8, inj_snr_9, inj_snr_10, inj_snr_11, inj_snr_12], axis=0)
    
    
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)


                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                    f13.close()
                    f14.close()
                
                elif(data_config.train.train_negative_latency_seconds == '10'):
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_1, 'r')
                    f2 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_2, 'r')
                    f3 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_3, 'r')
                    f4 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_4, 'r')
                    f5 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_5, 'r')
                    f6 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_6, 'r')
                    f7 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_7, 'r')
                    f8 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_8, 'r')
                    f9 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_9, 'r')
                    f10 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_10, 'r')
                    f11 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_11, 'r')
                    f12 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_12, 'r')
                    f13 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_13, 'r')
                    f14 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_14, 'r')
                    f15 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_15, 'r')
                    f16 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_16, 'r')
                    f17 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_17, 'r')
                    f18 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_18, 'r')
                    f19 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_19, 'r')
                    f20 = h5py.File(data_config.parameters.BNS.path_train_design_10_sec_20, 'r')
              
                    ra_1 = 2.0*np.pi*f1['ra'][0:50000][()]
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][0:50000][()])                    
                    mass_1_1 = f1['mass1'][0:50000][()]
                    mass_2_1 = f1['mass2'][0:50000][()]
                    spin_1_1 = f1['spin1z'][0:50000][()]
                    spin_2_1 = f1['spin2z'][0:50000][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                    inj_snr_1 = f1['Injection_SNR'][0:50000][()]
                                    
                    ra_2 = 2.0*np.pi*f2['ra'][0:50000][()]
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][0:50000][()])                   
                    mass_1_2 = f2['mass1'][0:50000][()]
                    mass_2_2 = f2['mass2'][0:50000][()]
                    spin_1_2 = f2['spin1z'][0:50000][()]
                    spin_2_2 = f2['spin2z'][0:50000][()]
#                    inc_2 = f2['inclination'][0:50000][()]
                    inj_snr_2 = f2['Injection_SNR'][0:50000][()]
                    
                    ra_3 = 2.0*np.pi*f3['ra'][0:50000][()]
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][0:50000][()])
                    mass_1_3 = f3['mass1'][0:50000][()]
                    mass_2_3 = f3['mass2'][0:50000][()]
                    spin_1_3 = f3['spin1z'][0:50000][()]
                    spin_2_3 = f3['spin2z'][0:50000][()]
#                    inc_3 = f3['inclination'][0:50000][()]
                    inj_snr_3 = f3['Injection_SNR'][0:50000][()]
                    
                    ra_4 = 2.0*np.pi*f4['ra'][()]
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                    mass_1_4 = f4['mass1'][()]
                    mass_2_4 = f4['mass2'][()]
                    spin_1_4 = f4['spin1z'][()]
                    spin_2_4 = f4['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_4 = f4['Injection_SNR'][()]
                    
                    ra_5 = 2.0*np.pi*f5['ra'][()]
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                    mass_1_5 = f5['mass1'][()]
                    mass_2_5 = f5['mass2'][()]
                    spin_1_5 = f5['spin1z'][()]
                    spin_2_5 = f5['spin2z'][()]
#                    inc_5 = f5['inclination'][()]
                    inj_snr_5 = f5['Injection_SNR'][()]
                    
                    ra_6 = 2.0*np.pi*f6['ra'][()]
                    dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                    mass_1_6 = f6['mass1'][()]
                    mass_2_6 = f6['mass2'][()]
                    spin_1_6 = f6['spin1z'][()]
                    spin_2_6 = f6['spin2z'][()]
#                    inc_6 = f6['inclination'][()]
                    inj_snr_6 = f6['Injection_SNR'][()]
                    
                    ra_7 = 2.0*np.pi*f7['ra'][()]
                    dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                    mass_1_7 = f7['mass1'][()]
                    mass_2_7 = f7['mass2'][()]
                    spin_1_7 = f7['spin1z'][()]
                    spin_2_7 = f7['spin2z'][()]
#                    inc_7 = f7['inclination'][()]
                    inj_snr_7 = f7['Injection_SNR'][()]
                    
                    ra_8 = 2.0*np.pi*f8['ra'][()]
                    dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                    mass_1_8 = f8['mass1'][()]
                    mass_2_8 = f8['mass2'][()]
                    spin_1_8 = f8['spin1z'][()]
                    spin_2_8 = f8['spin2z'][()]
#                    inc_8 = f8['inclination'][()]
                    inj_snr_8 = f8['Injection_SNR'][()]
                    
                    ra_9 = 2.0*np.pi*f9['ra'][()]
                    dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                    mass_1_9 = f9['mass1'][()]
                    mass_2_9 = f9['mass2'][()]
                    spin_1_9 = f9['spin1z'][()]
                    spin_2_9 = f9['spin2z'][()]
#                    inc_9 = f9['inclination'][()]
                    inj_snr_9 = f9['Injection_SNR'][()]
                    
                    ra_10 = 2.0*np.pi*f10['ra'][()]
                    dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
                    mass_1_10 = f10['mass1'][()]
                    mass_2_10 = f10['mass2'][()]
                    spin_1_10 = f10['spin1z'][()]
                    spin_2_10 = f10['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                    inj_snr_10 = f10['Injection_SNR'][()]
                    
                    ra_11 = 2.0*np.pi*f11['ra'][()]
                    dec_11 = np.arcsin(1.0-2.0*f11['dec'][()])
                    mass_1_11 = f11['mass1'][()]
                    mass_2_11 = f11['mass2'][()]
                    spin_1_11 = f11['spin1z'][()]
                    spin_2_11 = f11['spin2z'][()]
#                    inc_11 = f11['inclination'][()]
                    inj_snr_11 = f11['Injection_SNR'][()]
                    
                    ra_12 = 2.0*np.pi*f12['ra'][()]
                    dec_12 = np.arcsin(1.0-2.0*f12['dec'][()])
                    mass_1_12 = f12['mass1'][()]
                    mass_2_12 = f12['mass2'][()]
                    spin_1_12 = f12['spin1z'][()]
                    spin_2_12 = f12['spin2z'][()]
#                    inc_11 = f11['inclination'][()]
                    inj_snr_12 = f12['Injection_SNR'][()]
                    
                    ra_13 = 2.0*np.pi*f13['ra'][()]
                    dec_13 = np.arcsin(1.0-2.0*f13['dec'][()])
                    mass_1_13 = f13['mass1'][()]
                    mass_2_13 = f13['mass2'][()]
                    spin_1_13 = f13['spin1z'][()]
                    spin_2_13 = f13['spin2z'][()]
#                    inc_11 = f11['inclination'][()]
                    inj_snr_13 = f13['Injection_SNR'][()]
                    
                    ra_14 = 2.0*np.pi*f14['ra'][()]
                    dec_14 = np.arcsin(1.0-2.0*f14['dec'][()])
                    
                    ra_15 = 2.0*np.pi*f15['ra'][()]
                    dec_15 = f15['dec'][()]
                    
                    ra_16 = 2.0*np.pi*f16['ra'][()]
                    dec_16 = f16['dec'][()]
                    
                    ra_17 = 2.0*np.pi*f17['ra'][()]
                    dec_17 = f17['dec'][()]
                    
                    ra_18 = 2.0*np.pi*f18['ra'][()]
                    dec_18 = f18['dec'][()]
                    
                    ra_19 = 2.0*np.pi*f19['ra'][()]
                    dec_19 = f19['dec'][()]
                    
                    ra_20 = 2.0*np.pi*f20['ra'][()]
                    dec_20 = f20['dec'][()]
                
                    ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9, ra_10, ra_11, ra_12, ra_13], axis=0)                
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                
                    dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9, dec_10, dec_11, dec_12, dec_13], axis=0)

                    
                    mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9, mass_1_10, mass_1_11, mass_1_12, mass_1_13], axis=0)
                    mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9, mass_2_10, mass_2_11, mass_2_12, mass_2_13], axis=0)
                    spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9, spin_1_10, spin_1_11, spin_1_12, spin_1_13], axis=0)
                    spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9, spin_2_10, spin_2_11, spin_2_12, spin_2_13], axis=0)
 
                    inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_6, inj_snr_7, inj_snr_8, inj_snr_9, inj_snr_10, inj_snr_11, inj_snr_12, inj_snr_13], axis=0)
                                        
        
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                    f13.close()
                    f14.close()
                    f15.close()
                    f16.close()
                    f17.close()
                    f18.close()
                    f19.close()
                    f20.close()
                    
                elif(data_config.train.train_negative_latency_seconds == '15'):
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_1, 'r')
                    f2 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_2, 'r')
                    f3 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_3, 'r')
                    f4 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_4, 'r')
                    f5 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_5, 'r')
                    f6 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_6, 'r')
                    f7 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_7, 'r')
                    f8 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_8, 'r')
                    f9 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_9, 'r')
                    f10 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_10, 'r')
                    f11 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_11, 'r')
                    f12 = h5py.File(data_config.parameters.BNS.path_train_design_15_sec_12, 'r')
              
                    ra_1 = 2.0*np.pi*f1['ra'][()]
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])                    
                    mass_1_1 = f1['mass1'][()]
                    mass_2_1 = f1['mass2'][()]
                    spin_1_1 = f1['spin1z'][()]
                    spin_2_1 = f1['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_1 = f1['Injection_SNR'][()]
                
                    ra_2 = 2.0*np.pi*f2['ra'][()]
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                    mass_1_2 = f2['mass1'][()]
                    mass_2_2 = f2['mass2'][()]
                    spin_1_2 = f2['spin1z'][()]
                    spin_2_2 = f2['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_2 = f2['Injection_SNR'][()]
                    
                    ra_3 = 2.0*np.pi*f3['ra'][()]
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                    mass_1_3 = f3['mass1'][()]
                    mass_2_3 = f3['mass2'][()]
                    spin_1_3 = f3['spin1z'][()]
                    spin_2_3 = f3['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_3 = f3['Injection_SNR'][()]
                    
                    ra_4 = 2.0*np.pi*f4['ra'][()]
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                    mass_1_4 = f4['mass1'][()]
                    mass_2_4 = f4['mass2'][()]
                    spin_1_4 = f4['spin1z'][()]
                    spin_2_4 = f4['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_4 = f4['Injection_SNR'][()]
                    
                    ra_5 = 2.0*np.pi*f5['ra'][0:12000][()]  # 12000 
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][0:12000][()]) # 12000
                    mass_1_5 = f5['mass1'][0:12000][()] # 12000
                    mass_2_5 = f5['mass2'][0:12000][()] # 12000
                    spin_1_5 = f5['spin1z'][0:12000][()] # 12000
                    spin_2_5 = f5['spin2z'][0:12000][()] # 12000
#                    inc_4 = f4['inclination'][()]
                    inj_snr_5 = f5['Injection_SNR'][0:12000][()] # 12000
                    
                    ra_6 = 2.0*np.pi*f6['ra'][()]
                    dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                    mass_1_6 = f6['mass1'][()]
                    mass_2_6 = f6['mass2'][()]
                    spin_1_6 = f6['spin1z'][()]
                    spin_2_6 = f6['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_6 = f6['Injection_SNR'][()]
                    
                    ra_7 = 2.0*np.pi*f7['ra'][()]
                    dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                    mass_1_7 = f7['mass1'][()]
                    mass_2_7 = f7['mass2'][()]
                    spin_1_7 = f7['spin1z'][()]
                    spin_2_7 = f7['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_7 = f7['Injection_SNR'][()]
                    
                    ra_8 = 2.0*np.pi*f8['ra'][()]
                    dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                    mass_1_8 = f8['mass1'][()]
                    mass_2_8 = f8['mass2'][()]
                    spin_1_8 = f8['spin1z'][()]
                    spin_2_8 = f8['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_8 = f8['Injection_SNR'][()]
                    
                    ra_9 = 2.0*np.pi*f9['ra'][()]
                    dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                    mass_1_9 = f9['mass1'][()]
                    mass_2_9 = f9['mass2'][()]
                    spin_1_9 = f9['spin1z'][()]
                    spin_2_9 = f9['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_9 = f9['Injection_SNR'][()]
    
                    ra_10 = 2.0*np.pi*f10['ra'][()]
                    dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
                    mass_1_10 = f10['mass1'][()]
                    mass_2_10 = f10['mass2'][()]
                    spin_1_10 = f10['spin1z'][()]
                    spin_2_10 = f10['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_10 = f10['Injection_SNR'][()]
    
                    ra_11 = 2.0*np.pi*f11['ra'][()]
                    dec_11 = np.arcsin(1.0-2.0*f11['dec'][()])
                    mass_1_11 = f11['mass1'][()]
                    mass_2_11 = f11['mass2'][()]
                    spin_1_11 = f11['spin1z'][()]
                    spin_2_11 = f11['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_11 = f11['Injection_SNR'][()]
    
                    ra_12 = 2.0*np.pi*f12['ra'][()]
                    dec_12 = np.arcsin(1.0-2.0*f12['dec'][()])
                    mass_1_12 = f12['mass1'][()]
                    mass_2_12 = f12['mass2'][()]
                    spin_1_12 = f12['spin1z'][()]
                    spin_2_12 = f12['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_12 = f12['Injection_SNR'][()]
                
                    ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9, ra_10, ra_11, ra_12], axis=0)
                
                    dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9, dec_10, dec_11, dec_12], axis=0)
                    
                    
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                    
                
                    mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9, mass_1_10, mass_1_11, mass_1_12], axis=0)
                    mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9, mass_2_10, mass_2_11, mass_2_12], axis=0)
                    spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9, spin_1_10, spin_1_11, spin_1_12], axis=0)
                    spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9, spin_2_10, spin_2_11, spin_2_12], axis=0)

                    inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_6, inj_snr_7, inj_snr_8, inj_snr_9, inj_snr_10, inj_snr_11, inj_snr_12], axis=0)
    
        
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()  
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                    
                elif(data_config.train.train_negative_latency_seconds == '30'):
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_1, 'r')
                    f2 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_2, 'r')
                    f3 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_3, 'r')
                    f4 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_4, 'r')
                    f5 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_5, 'r')
                    f6 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_6, 'r')
                    f7 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_7, 'r')
                    f8 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_8, 'r')
                    f9 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_9, 'r')
                    f10 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_10, 'r')
                    f11 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_11, 'r')
                    f12 = h5py.File(data_config.parameters.BNS.path_train_design_30_sec_12, 'r')                    
              
                    ra_1 = 2.0*np.pi*f1['ra'][()]
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])                    
                    mass_1_1 = f1['mass1'][()]
                    mass_2_1 = f1['mass2'][()]
                    spin_1_1 = f1['spin1z'][()]
                    spin_2_1 = f1['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_1 = f1['Injection_SNR'][()]
                
                    ra_2 = 2.0*np.pi*f2['ra'][()]
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                    mass_1_2 = f2['mass1'][()]
                    mass_2_2 = f2['mass2'][()]
                    spin_1_2 = f2['spin1z'][()]
                    spin_2_2 = f2['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_2 = f2['Injection_SNR'][()]
                    
                    ra_3 = 2.0*np.pi*f3['ra'][()]
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                    mass_1_3 = f3['mass1'][()]
                    mass_2_3 = f3['mass2'][()]
                    spin_1_3 = f3['spin1z'][()]
                    spin_2_3 = f3['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_3 = f3['Injection_SNR'][()]
                    
                    ra_4 = 2.0*np.pi*f4['ra'][()]
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                    mass_1_4 = f4['mass1'][()]
                    mass_2_4 = f4['mass2'][()]
                    spin_1_4 = f4['spin1z'][()]
                    spin_2_4 = f4['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_4 = f4['Injection_SNR'][()]
                    
                    ra_5 = 2.0*np.pi*f5['ra'][0:26000][()]
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][0:26000][()])
                    mass_1_5 = f5['mass1'][0:26000][()]
                    mass_2_5 = f5['mass2'][0:26000][()]
                    spin_1_5 = f5['spin1z'][0:26000][()]
                    spin_2_5 = f5['spin2z'][0:26000][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_5 = f5['Injection_SNR'][0:26000][()]
                    
                    ra_6 = 2.0*np.pi*f6['ra'][()]
                    dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                    mass_1_6 = f6['mass1'][()]
                    mass_2_6 = f6['mass2'][()]
                    spin_1_6 = f6['spin1z'][()]
                    spin_2_6 = f6['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_6 = f6['Injection_SNR'][()]
                    
                    ra_7 = 2.0*np.pi*f7['ra'][()]
                    dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                    mass_1_7 = f7['mass1'][()]
                    mass_2_7 = f7['mass2'][()]
                    spin_1_7 = f7['spin1z'][()]
                    spin_2_7 = f7['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_7 = f7['Injection_SNR'][()]
                    
                    ra_8 = 2.0*np.pi*f8['ra'][()]
                    dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                    mass_1_8 = f8['mass1'][()]
                    mass_2_8 = f8['mass2'][()]
                    spin_1_8 = f8['spin1z'][()]
                    spin_2_8 = f8['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_8 = f8['Injection_SNR'][()]
                    
                    ra_9 = 2.0*np.pi*f9['ra'][()]
                    dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                    mass_1_9 = f9['mass1'][()]
                    mass_2_9 = f9['mass2'][()]
                    spin_1_9 = f9['spin1z'][()]
                    spin_2_9 = f9['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_9 = f9['Injection_SNR'][()]
    
                    ra_10 = 2.0*np.pi*f10['ra'][()]
                    dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
                    mass_1_10 = f10['mass1'][()]
                    mass_2_10 = f10['mass2'][()]
                    spin_1_10 = f10['spin1z'][()]
                    spin_2_10 = f10['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_10 = f10['Injection_SNR'][()]
    
                    ra_11 = 2.0*np.pi*f11['ra'][()]
                    dec_11 = np.arcsin(1.0-2.0*f11['dec'][()])
                    mass_1_11 = f11['mass1'][()]
                    mass_2_11 = f11['mass2'][()]
                    spin_1_11 = f11['spin1z'][()]
                    spin_2_11 = f11['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_11 = f11['Injection_SNR'][()]
    
                    ra_12 = 2.0*np.pi*f12['ra'][()]
                    dec_12 = np.arcsin(1.0-2.0*f12['dec'][()])
                    mass_1_12 = f12['mass1'][()]
                    mass_2_12 = f12['mass2'][()]
                    spin_1_12 = f12['spin1z'][()]
                    spin_2_12 = f12['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_12 = f12['Injection_SNR'][()]
                
                    ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9, ra_10, ra_11, ra_12], axis=0)
                
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                
                    dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9, dec_10, dec_11, dec_12], axis=0)
                    
                    mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9, mass_1_10, mass_1_11, mass_1_12], axis=0)
                    mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9, mass_2_10, mass_2_11, mass_2_12], axis=0)
                    spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9, spin_1_10, spin_1_11, spin_1_12], axis=0)
                    spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9, spin_2_10, spin_2_11, spin_2_12], axis=0)
#                    inc = np.concatenate([inc_1, inc_2, inc_3, inc_4, inc_5, inc_6, inc_7, inc_8, inc_9, inc_10, inc_11], axis=0)
                    inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_6, inj_snr_7, inj_snr_8, inj_snr_9, inj_snr_10, inj_snr_11, inj_snr_12], axis=0)
        
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()  
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close()
                
                elif(data_config.train.train_negative_latency_seconds == '45'):
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_1, 'r')
                    f2 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_2, 'r')
                    f3 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_3, 'r')
                    f4 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_4, 'r')
                    f5 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_5, 'r')
                    f6 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_6, 'r')
                    f7 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_7, 'r')
                    f8 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_8, 'r')
                    f9 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_9, 'r')
                    f10 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_10, 'r')
                    f11 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_11, 'r')
                    f12 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_12, 'r')
                    f13 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_13, 'r')
                    f14 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_14, 'r')
                    f15 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_15, 'r')
                    f16 = h5py.File(data_config.parameters.BNS.path_train_design_45_sec_16, 'r')
              
                    ra_1 = 2.0*np.pi*f1['ra'][()]
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])                    
                    mass_1_1 = f1['mass1'][()]
                    mass_2_1 = f1['mass2'][()]
                    spin_1_1 = f1['spin1z'][()]
                    spin_2_1 = f1['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_1 = f1['Injection_SNR'][()]
                
                    ra_2 = 2.0*np.pi*f2['ra'][()]
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                    mass_1_2 = f2['mass1'][()]
                    mass_2_2 = f2['mass2'][()]
                    spin_1_2 = f2['spin1z'][()]
                    spin_2_2 = f2['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_2 = f2['Injection_SNR'][()]
                    
                    ra_3 = 2.0*np.pi*f3['ra'][()]
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                    mass_1_3 = f3['mass1'][()]
                    mass_2_3 = f3['mass2'][()]
                    spin_1_3 = f3['spin1z'][()]
                    spin_2_3 = f3['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_3 = f3['Injection_SNR'][()]
                    
                    ra_4 = 2.0*np.pi*f4['ra'][()]
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                    mass_1_4 = f4['mass1'][()]
                    mass_2_4 = f4['mass2'][()]
                    spin_1_4 = f4['spin1z'][()]
                    spin_2_4 = f4['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_4 = f4['Injection_SNR'][()]
    
                    ra_5 = 2.0*np.pi*f5['ra'][()]
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                    mass_1_5 = f5['mass1'][()]
                    mass_2_5 = f5['mass2'][()]
                    spin_1_5 = f5['spin1z'][()]
                    spin_2_5 = f5['spin2z'][()]
#                    inc_5 = f5['inclination'][()]
                    inj_snr_5 = f5['Injection_SNR'][()]
                    
                    ra_5 = 2.0*np.pi*f5['ra'][0:42000][()]
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][0:42000][()])
                    mass_1_5 = f5['mass1'][0:42000][()]
                    mass_2_5 = f5['mass2'][0:42000][()]
                    spin_1_5 = f5['spin1z'][0:42000][()]
                    spin_2_5 = f5['spin2z'][0:42000][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_5 = f5['Injection_SNR'][0:42000][()]
                    
                    ra_6 = 2.0*np.pi*f6['ra'][()]
                    dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                    mass_1_6 = f6['mass1'][()]
                    mass_2_6 = f6['mass2'][()]
                    spin_1_6 = f6['spin1z'][()]
                    spin_2_6 = f6['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_6 = f6['Injection_SNR'][()]
                    
                    ra_7 = 2.0*np.pi*f7['ra'][()]
                    dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                    mass_1_7 = f7['mass1'][()]
                    mass_2_7 = f7['mass2'][()]
                    spin_1_7 = f7['spin1z'][()]
                    spin_2_7 = f7['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_7 = f7['Injection_SNR'][()]
                    
                    ra_8 = 2.0*np.pi*f8['ra'][()]
                    dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                    mass_1_8 = f8['mass1'][()]
                    mass_2_8 = f8['mass2'][()]
                    spin_1_8 = f8['spin1z'][()]
                    spin_2_8 = f8['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_8 = f8['Injection_SNR'][()]
                    
                    ra_9 = 2.0*np.pi*f9['ra'][()]
                    dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                    mass_1_9 = f9['mass1'][()]
                    mass_2_9 = f9['mass2'][()]
                    spin_1_9 = f9['spin1z'][()]
                    spin_2_9 = f9['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_9 = f9['Injection_SNR'][()]
    
                    ra_10 = 2.0*np.pi*f10['ra'][()]
                    dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
                    mass_1_10 = f10['mass1'][()]
                    mass_2_10 = f10['mass2'][()]
                    spin_1_10 = f10['spin1z'][()]
                    spin_2_10 = f10['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_10 = f10['Injection_SNR'][()]
    
                    ra_11 = 2.0*np.pi*f11['ra'][()]
                    dec_11 = np.arcsin(1.0-2.0*f11['dec'][()])
                    mass_1_11 = f11['mass1'][()]
                    mass_2_11 = f11['mass2'][()]
                    spin_1_11 = f11['spin1z'][()]
                    spin_2_11 = f11['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_11 = f11['Injection_SNR'][()]
    
                    ra_12 = 2.0*np.pi*f12['ra'][()]
                    dec_12 = np.arcsin(1.0-2.0*f12['dec'][()])
                    mass_1_12 = f12['mass1'][()]
                    mass_2_12 = f12['mass2'][()]
                    spin_1_12 = f12['spin1z'][()]
                    spin_2_12 = f12['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_12 = f12['Injection_SNR'][()]
                    
                    ra_13 = 2.0*np.pi*f13['ra'][()]
                    dec_13 = np.arcsin(1.0-2.0*f13['dec'][()])
                    mass_1_13 = f13['mass1'][()]
                    mass_2_13 = f13['mass2'][()]
                    spin_1_13 = f13['spin1z'][()]
                    spin_2_13 = f13['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_13 = f13['Injection_SNR'][()]
                    
                    ra_14 = 2.0*np.pi*f14['ra'][()]
                    dec_14 = np.arcsin(1.0-2.0*f14['dec'][()])
                    mass_1_14 = f14['mass1'][()]
                    mass_2_14 = f14['mass2'][()]
                    spin_1_14 = f14['spin1z'][()]
                    spin_2_14 = f14['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_14 = f14['Injection_SNR'][()]
    
                    ra_15 = 2.0*np.pi*f15['ra'][()]
                    dec_15 = np.arcsin(1.0-2.0*f15['dec'][()])
                    mass_1_15 = f15['mass1'][()]
                    mass_2_15 = f15['mass2'][()]
                    spin_1_15 = f15['spin1z'][()]
                    spin_2_15 = f15['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_15 = f15['Injection_SNR'][()]
    
                    ra_16 = 2.0*np.pi*f16['ra'][()]
                    dec_16 = np.arcsin(1.0-2.0*f16['dec'][()])
                    mass_1_16 = f16['mass1'][()]
                    mass_2_16 = f16['mass2'][()]
                    spin_1_16 = f16['spin1z'][()]
                    spin_2_16 = f16['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_16 = f16['Injection_SNR'][()]
    
                
                    ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9, ra_10, ra_11, ra_12, ra_13, ra_14, ra_15, ra_16], axis=0)
                
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                
                    dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9, dec_10, dec_11, dec_12, dec_13, dec_14, dec_15, dec_16], axis=0)
                    
                    mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9, mass_1_10, mass_1_11, mass_1_12, mass_1_13, mass_1_14, mass_1_15, mass_1_16], axis=0)
                    mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9, mass_2_10, mass_2_11, mass_2_12, mass_2_13, mass_2_14, mass_2_15, mass_2_16], axis=0)
                    spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5,  spin_1_6, spin_1_7, spin_1_8, spin_1_9, spin_1_10, spin_1_11, spin_1_12, spin_1_13, spin_1_14, spin_1_15, spin_1_16], axis=0)
                    spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9, spin_2_10, spin_2_11, spin_2_12, spin_2_13, spin_2_14, spin_2_15, spin_2_16], axis=0)
#                    inc = np.concatenate([inc_1, inc_2, inc_3, inc_4, inc_5, inc_6, inc_7, inc_8, inc_9, inc_10, inc_11], axis=0)
                    inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_6, inj_snr_7, inj_snr_8, inj_snr_9, inj_snr_10, inj_snr_11, inj_snr_12, inj_snr_13, inj_snr_14, inj_snr_15, inj_snr_16], axis=0)
        
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()  
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                    f10.close()
                    f11.close()
                    f12.close() 
                    f13.close()
                    f14.close()
                    f15.close()
                    f16.close()
                    
                 
                elif(data_config.train.train_negative_latency_seconds == '58'):
                    
                    f1 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_1, 'r')
                    f2 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_2, 'r')
                    f3 = h5py.File(data_config.parameters.parameters.BNS.path_train_design_58_sec_3, 'r')
                    f4 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_4, 'r')
                    f5 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_5, 'r')
                    f6 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_6, 'r')
                    f7 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_7, 'r')
                    f8 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_8, 'r')
                    f9 = h5py.File(data_config.parameters.BNS.path_train_design_58_sec_9, 'r')
              
                    ra_1 = 2.0*np.pi*f1['ra'][()]
                    dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])                    
                    mass_1_1 = f1['mass1'][()]
                    mass_2_1 = f1['mass2'][()]
                    spin_1_1 = f1['spin1z'][()]
                    spin_2_1 = f1['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_1 = f1['Injection_SNR'][()]
                
                    ra_2 = 2.0*np.pi*f2['ra'][()]
                    dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                    mass_1_2 = f2['mass1'][()]
                    mass_2_2 = f2['mass2'][()]
                    spin_1_2 = f2['spin1z'][()]
                    spin_2_2 = f2['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_2 = f2['Injection_SNR'][()]
                    
                    ra_3 = 2.0*np.pi*f3['ra'][()]
                    dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                    mass_1_3 = f3['mass1'][()]
                    mass_2_3 = f3['mass2'][()]
                    spin_1_3 = f3['spin1z'][()]
                    spin_2_3 = f3['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_3 = f3['Injection_SNR'][()]
                    
                    ra_4 = 2.0*np.pi*f4['ra'][0:62000][()]
                    dec_4 = np.arcsin(1.0-2.0*f4['dec'][0:62000][()])
                    mass_1_4 = f4['mass1'][0:62000][()]
                    mass_2_4 = f4['mass2'][0:62000][()]
                    spin_1_4 = f4['spin1z'][0:62000][()]
                    spin_2_4 = f4['spin2z'][0:62000][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_4 = f4['Injection_SNR'][0:62000][()]
                    
                    ra_5 = 2.0*np.pi*f5['ra'][()]
                    dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                    mass_1_5 = f5['mass1'][()]
                    mass_2_5 = f5['mass2'][()]
                    spin_1_5 = f5['spin1z'][()]
                    spin_2_5 = f5['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_5 = f5['Injection_SNR'][()]
                    
                    ra_6 = 2.0*np.pi*f6['ra'][()]
                    dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                    mass_1_6 = f6['mass1'][()]
                    mass_2_6 = f6['mass2'][()]
                    spin_1_6 = f6['spin1z'][()]
                    spin_2_6 = f6['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_6 = f6['Injection_SNR'][()]
                    
                    ra_7 = 2.0*np.pi*f7['ra'][()]
                    dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                    mass_1_7 = f7['mass1'][()]
                    mass_2_7 = f7['mass2'][()]
                    spin_1_7 = f7['spin1z'][()]
                    spin_2_7 = f7['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_7 = f7['Injection_SNR'][()]
                    
                    ra_8 = 2.0*np.pi*f8['ra'][()]
                    dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                    mass_1_8 = f8['mass1'][()]
                    mass_2_8 = f8['mass2'][()]
                    spin_1_8 = f8['spin1z'][()]
                    spin_2_8 = f8['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_8 = f8['Injection_SNR'][()]
                    
                    ra_9 = 2.0*np.pi*f9['ra'][()]
                    dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                    mass_1_9 = f9['mass1'][()]
                    mass_2_9 = f9['mass2'][()]
                    spin_1_9 = f9['spin1z'][()]
                    spin_2_9 = f9['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                    inj_snr_9 = f9['Injection_SNR'][()]
                
                    ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7, ra_8, ra_9], axis=0)
                
                    ra = ra - np.pi
                    ra_x = np.cos(ra)
                    ra_y = np.sin(ra)
                
                    dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7, dec_8, dec_9], axis=0)
                    
                    mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7, mass_1_8, mass_1_9], axis=0)
                    mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7, mass_2_8, mass_2_9], axis=0)
                    spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7, spin_1_8, spin_1_9], axis=0)
                    spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7, spin_2_8, spin_2_9], axis=0)
#                    inc = np.concatenate([inc_1, inc_2, inc_3, inc_4, inc_5, inc_6, inc_7, inc_8, inc_9, inc_10, inc_11], axis=0)
                    inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5, inj_snr_6, inj_snr_7, inj_snr_8, inj_snr_9], axis=0)
        
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close() 
                    f5.close()
                    f6.close()
                    f7.close()
                    f8.close()
                    f9.close()
                
                
                
            elif((data_config.train.train_real == True) and (data_config.train.train_negative_latency == False) and ((data_config.train.PSD == 'O2'))):
                    
                f1 = h5py.File(data_config.parameters.BNS.path_train_O2_noise_1, 'r')
                f2 = h5py.File(data_config.parameters.BNS.path_train_O2_noise_2, 'r')
                f3 = h5py.File(data_config.parameters.NS.path_train_O2_noise_3, 'r')
                f4 = h5py.File(data_config.parameters.BNS.path_train_O2_noise_4, 'r')
                f5 = h5py.File(data_config.parameters.BNS.path_train_O2_noise_5, 'r')
            
                ra_1 = 2.0*np.pi*f1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                mass_1_1 = f1['mass1'][()]
                mass_2_1 = f1['mass2'][()]
                spin_1_1 = f1['spin1z'][()]
                spin_2_1 = f1['spin2z'][()]
#                    inc_1 = f1['inclination'][0:50000][()]
                inj_snr_1 = f1['Injection_SNR'][()]
        
                ra_2 = 2.0*np.pi*f2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                mass_1_2 = f2['mass1'][()]
                mass_2_2 = f2['mass2'][()]
                spin_1_2 = f2['spin1z'][()]
                spin_2_2 = f2['spin2z'][()]
#                   inc_2 = f2['inclination'][0:50000][()]
                inj_snr_2 = f2['Injection_SNR'][()]
                
                ra_3 = 2.0*np.pi*f3['ra'][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                mass_1_3 = f3['mass1'][()]
                mass_2_3 = f3['mass2'][()]
                spin_1_3 = f3['spin1z'][()]
                spin_2_3 = f3['spin2z'][()]
#                    inc_3 = f3['inclination'][0:50000][()]
                inj_snr_3 = f3['Injection_SNR'][()]
                    
                ra_4 = 2.0*np.pi*f4['ra'][()]
                dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                mass_1_4 = f4['mass1'][()]
                mass_2_4 = f4['mass2'][()]
                spin_1_4 = f4['spin1z'][()]
                spin_2_4 = f4['spin2z'][()]
#                    inc_4 = f4['inclination'][()]
                inj_snr_4 = f4['Injection_SNR'][()]
                    
                ra_5 = 2.0*np.pi*f5['ra'][()]
                dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                mass_1_5 = f5['mass1'][()]
                mass_2_5 = f5['mass2'][()]
                spin_1_5 = f5['spin1z'][()]
                spin_2_5 = f5['spin2z'][()]
#                    inc_5 = f5['inclination'][()]
                inj_snr_5 = f5['Injection_SNR'][()]
                    
    
                ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5], axis=0)
                dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5], axis=0)
                mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_1_3, mass_1_4, mass_1_5], axis=0)
                mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5], axis=0)
                spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5], axis=0)
                spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5], axis=0)
                inj_snr = np.concatenate([inj_snr_1, inj_snr_2, inj_snr_3, inj_snr_4, inj_snr_5], axis=0)
    
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)

                
                f1.close()
                f2.close()
                f3.close()
                f4.close()
                f5.close()
                
                             
        ra = ra[:,None]
        ra_x = ra_x[:,None]
        ra_y = ra_y[:,None]
        
        dec = dec[:,None]
        
        mass_1 = mass_1[:,None]
        mass_2 = mass_2[:,None]
        spin_1 = spin_1[:,None]
        spin_2 = spin_2[:,None]

        y_train = np.concatenate((ra_x, ra_y, dec), axis=1).astype('float32')
          
        intrinsic_params = np.concatenate((mass_1, mass_2, spin_1, spin_2), axis=1)

        return y_train, intrinsic_params
   
    @staticmethod
    def load_train_2_det_parameters(data_config):
        """Loads train parameters from path"""
        if(data_config.train.train_negative_latency == False):
            if((data_config.train.dataset == 'BNS') and (data_config.train.snr_range_train == 'low')):
            
                f1 = h5py.File(data_config.parameters.BNS.path_train_2_det_low_SNR, 'r')
                f2 = h5py.File(data_config.parameters.BNS.path_train_low_snr_1, 'r')
                f3 = h5py.File(data_config.parameters.BNS.path_train_low_snr_2, 'r')
                f4 = h5py.File(data_config.parameters.BNS.path_train_low_snr_3, 'r')
                f5 = h5py.File(data_config.parameters.BNS.path_train_1, 'r')
                f6 = h5py.File(data_config.parameters.BNS.path_train_2, 'r')
                f7 = h5py.File(data_config.parameters.BNS.path_train_2_det_low_SNR_1, 'r')
                f8 = h5py.File(data_config.parameters.BNS.path_train_2_det_high_SNR_1, 'r')
                f9 = h5py.File(data_config.parameters.BNS.path_train_2_det_high_SNR_2, 'r')
            
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
            
            elif((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_train == 'low')):
            
                f1 = h5py.File(data_config.parameters.NSBH.path_train_1, 'r')
                f2 = h5py.File(data_config.parameters.NSBH.path_train_2, 'r')
                f3 = h5py.File(data_config.parameters.NSBH.path_train_3, 'r')
                f4 = h5py.File(data_config.parameters.NSBH.path_train_4, 'r')
                f5 = h5py.File(data_config.parameters.NSBH.path_train_low_snr_1, 'r')
                f6 = h5py.File(data_config.parameters.NSBH.path_train_low_snr_2, 'r')
                f7 = h5py.File(data_config.parameters.NSBH.path_train_low_snr_3, 'r')
        
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
            
            elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == False) and (data_config.train.snr_range_train == 'low')):
            
                f1 = h5py.File(data_config.parameters.BBH.path_train, 'r')
                f2 = h5py.File(data_config.parameters.BBH.path_train_low_SNR, 'r')

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
            
            elif((data_config.train.dataset == 'BBH') and (data_config.train.train_real == True) and (data_config.train.PSD == 'O2')):
                
                f1 = h5py.File(data_config.parameters.BBH.path_train_real_noise_1, 'r')
                f2 = h5py.File(data_config.parameters.BBH.path_train_real_noise_2, 'r')
                f3 = h5py.File(data_config.parameters.BBH.path_train_real_noise_3, 'r')
                f4 = h5py.File(data_config.parameters.BBH.path_train_real_noise_4, 'r')
                f5 = h5py.File(data_config.parameters.BBH.path_train_real_noise_5, 'r')
                f6 = h5py.File(data_config.parameters.BBH.path_train_real_noise_6, 'r')
                f7 = h5py.File(data_config.parameters.BBH.path_train_real_noise_7, 'r')
                
                f8 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_1, 'r')
                f9 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_2, 'r')
                f10 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_3, 'r')
                f11 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_4, 'r')
                f12 = h5py.File(data_config.parameters.NSBH.path_train_real_noise_5, 'r')
                
                ra_1 = 2.0*np.pi*f1['ra'][()]
                dec_1 = np.arcsin(1.0-2.0*f1['dec'][()])
                mass_1_1 = f1['mass1'][()]
                mass_2_1 = f1['mass2'][()]
                spin_1_1 = f1['spin1z'][()]
                spin_2_1 = f1['spin2z'][()]
                gps_time_1 = f1['gps_time'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
    #                inj_snr_1 = f1['Injection_SNR'][()]
                
                ra_2 = 2.0*np.pi*f2['ra'][()]
                dec_2 = np.arcsin(1.0-2.0*f2['dec'][()])
                mass_1_2 = f2['mass1'][()]
                mass_2_2 = f2['mass2'][()]
                spin_1_2 = f2['spin1z'][()]
                spin_2_2 = f2['spin2z'][()]
                gps_time_2 = f2['gps_time'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
    #                inj_snr_2 = f2['Injection_SNR'][()]
    
                ra_3 = 2.0*np.pi*f3['ra'][()]
                dec_3 = np.arcsin(1.0-2.0*f3['dec'][()])
                mass_1_3 = f3['mass1'][()]
                mass_2_3 = f3['mass2'][()]
                spin_1_3 = f3['spin1z'][()]
                spin_2_3 = f3['spin2z'][()]
                gps_time_3 = f3['gps_time'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
    #                inj_snr_3 = f3['Injection_SNR'][()]
    
                ra_4 = 2.0*np.pi*f4['ra'][()]
                dec_4 = np.arcsin(1.0-2.0*f4['dec'][()])
                mass_1_4 = f4['mass1'][()]
                mass_2_4 = f4['mass2'][()]
                spin_1_4 = f4['spin1z'][()]
                spin_2_4 = f4['spin2z'][()]
                gps_time_4 = f4['gps_time'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
    #                inj_snr_4 = f4['Injection_SNR'][()]
    
                ra_5 = 2.0*np.pi*f5['ra'][()]
                dec_5 = np.arcsin(1.0-2.0*f5['dec'][()])
                mass_1_5 = f5['mass1'][()]
                mass_2_5 = f5['mass2'][()]
                spin_1_5 = f5['spin1z'][()]
                spin_2_5 = f5['spin2z'][()]
                gps_time_5 = f5['gps_time'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
    #                inj_snr_5 = f5['Injection_SNR'][()]

                ra_6 = 2.0*np.pi*f6['ra'][()]
                dec_6 = np.arcsin(1.0-2.0*f6['dec'][()])
                mass_1_6 = f6['mass1'][()]
                mass_2_6 = f6['mass2'][()]
                spin_1_6 = f6['spin1z'][()]
                spin_2_6 = f6['spin2z'][()]
                gps_time_6 = f6['gps_time'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
    #                inj_snr_5 = f5['Injection_SNR'][()]


                ra_7 = 2.0*np.pi*f7['ra'][()]
                dec_7 = np.arcsin(1.0-2.0*f7['dec'][()])
                mass_1_7 = f7['mass1'][()]
                mass_2_7 = f7['mass2'][()]
                spin_1_7 = f7['spin1z'][()]
                spin_2_7 = f7['spin2z'][()]
                gps_time_7 = f7['gps_time'][()]
    #                    inc_1 = f1['inclination'][0:50000][()]
    #                inj_snr_4 = f4['Injection_SNR'][()]
    
                ra_8 = 2.0*np.pi*f8['ra'][()]
                dec_8 = np.arcsin(1.0-2.0*f8['dec'][()])
                mass_1_8 = f8['mass1'][()]
                mass_2_8 = f8['mass2'][()]
                spin_1_8 = f8['spin1z'][()]
                spin_2_8 = f8['spin2z'][()]
#                    inc_8 = f8['inclination'][()]
                gps_time_8 = f8['gps_time'][()]
                    
                ra_9 = 2.0*np.pi*f9['ra'][()]
                dec_9 = np.arcsin(1.0-2.0*f9['dec'][()])
                mass_1_9 = f9['mass1'][()]
                mass_2_9 = f9['mass2'][()]
                spin_1_9 = f9['spin1z'][()]
                spin_2_9 = f9['spin2z'][()]
#                   inc_9 = f9['inclination'][()]
                gps_time_9 = f9['gps_time'][()]
                    
                ra_10 = 2.0*np.pi*f10['ra'][()]
                dec_10 = np.arcsin(1.0-2.0*f10['dec'][()])
                mass_1_10 = f10['mass1'][()]
                mass_2_10 = f10['mass2'][()]
                spin_1_10 = f10['spin1z'][()]
                spin_2_10 = f10['spin2z'][()]
#                    inc_10 = f10['inclination'][()]
                gps_time_10 = f10['gps_time'][()]
                    
                ra_11 = 2.0*np.pi*f11['ra'][()]
                dec_11 = np.arcsin(1.0-2.0*f11['dec'][()])
                mass_1_11 = f11['mass1'][()]
                mass_2_11 = f11['mass2'][()]
                spin_1_11 = f11['spin1z'][()]
                spin_2_11 = f11['spin2z'][()]
#                    inc_11 = f11['inclination'][()]
                gps_time_11 = f11['gps_time'][()]
                    
                ra_12 = 2.0*np.pi*f12['ra'][()]
                dec_12 = np.arcsin(1.0-2.0*f12['dec'][()])
                mass_1_12 = f12['mass1'][()]
                mass_2_12 = f12['mass2'][()]
                spin_1_12 = f12['spin1z'][()]
                spin_2_12 = f12['spin2z'][()]
#                    inc_11 = f11['inclination'][()]
                gps_time_12 = f12['gps_time'][()]                
        
                ra = np.concatenate([ra_1, ra_2, ra_3, ra_4, ra_5, ra_6, ra_7], axis=0)
                dec = np.concatenate([dec_1, dec_2, dec_3, dec_4, dec_5, dec_6, dec_7], axis=0)
                mass_1 = np.concatenate([mass_1_1, mass_1_2, mass_2_3, mass_1_4, mass_1_5, mass_1_6, mass_1_7], axis=0)
                mass_2 = np.concatenate([mass_2_1, mass_2_2, mass_2_3, mass_2_4, mass_2_5, mass_2_6, mass_2_7], axis=0)
                spin_1 = np.concatenate([spin_1_1, spin_1_2, spin_1_3, spin_1_4, spin_1_5, spin_1_6, spin_1_7], axis=0)
                spin_2 = np.concatenate([spin_2_1, spin_2_2, spin_2_3, spin_2_4, spin_2_5, spin_2_6, spin_2_7], axis=0)
                gps_time = np.concatenate([gps_time_1, gps_time_2, gps_time_3, gps_time_4, gps_time_5, gps_time_6, gps_time_7], axis=0)
            
        
                ra = ra - np.pi
                ra_x = np.cos(ra)
                ra_y = np.sin(ra)
            
            
        elif((data_config.train.train_negative_latency == True)):
            
            if((data_config.train.dataset == 'BNS') and (data_config.train.train_negative_latency_seconds == '5')):
                
                f1 = h5py.File(data_config.parameters.BNS.path_train_2_det_negative_latency_5_1, 'r')
                f2 = h5py.File(data_config.parameters.BNS.path_train_2_det_negative_latency_5_2, 'r')
            
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
            
            elif((data_config.train.dataset == 'BNS') and (data_config.train.train_negative_latency_seconds == '10')):
                
                f1 = h5py.File(data_config.parameters.BNS.path_train_2_det_negative_latency_10, 'r')
            
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
        
        mass_1 = mass_1[:,None]
        mass_2 = mass_2[:,None]
        spin_1 = spin_1[:,None]
        spin_2 = spin_2[:,None]

        y_train = np.concatenate((ra_x, ra_y, dec), axis=1).astype('float32')

        intrinsic_params = np.concatenate((mass_1, mass_2, spin_1, spin_2), axis=1)

        return y_train, intrinsic_params
            
    @staticmethod
    def load_test_3_det_data(data_config):
        """Loads dataset from path"""
        #Get the HDF5 group
        if(data_config.train.test_negative_latency == False):
        #NSBH
            if(data_config.train.dataset == 'NSBH'):
                if(data_config.train.snr_range_test == 'high'):
                    f_test = h5py.File(data_config.data.NSBH.path_test, 'r')
                    
                if((data_config.train.test_real == True) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                    
                    f1 = h5py.File(data_config.data.NSBH.path_test_GW190917, 'r')
                    
                    h1_test_real = np.real(f1['h1_snr_series'][()])
                    l1_test_real = np.real(f1['l1_snr_series'][()])
                    v1_test_real = np.real(f1['v1_snr_series'][()])
                    
                    h1_test_imag = np.realnp.imag((f1['h1_snr_series'][()]))
                    l1_test_imag = np.realnp.imag((f1['l1_snr_series'][()]))
                    v1_test_imag = np.realnp.imag((f1['v1_snr_series'][()]))
                    
                    f1.close()
                
                
                elif((data_config.train.test_real == True) and (data_config.train.PSD == 'O2')):
                    
                    f1 = h5py.File(data_config.data.NSBH.path_test_GW190814, 'r')
                   
                    h1_real_1 = np.real(f1['h1_snr_series'][()])
                    l1_real_1 = np.real(f1['l1_snr_series'][()])
                    v1_real_1 = np.real(f1['v1_snr_series'][()])
                    
                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
        
                    h1_real_1 = h1_real_1[None,:]
                    l1_real_1 = l1_real_1[None,:]                    
                    v1_real_1 = v1_real_1[None,:]                    

                    h1_imag_1 = h1_imag_1[None,:]                   
                    l1_imag_1 = l1_imag_1[None,:]                    
                    v1_imag_1 = v1_imag_1[None,:] 
                    
                    h1_test_real = h1_real_1
                    l1_test_real = l1_real_1
                    v1_test_real = v1_real_1
                    
                    h1_test_imag = h1_imag_1
                    l1_test_imag = l1_imag_1
                    v1_test_imag = v1_imag_1                    
            
                    f1.close()
                
                
                elif((data_config.train.test_real == False) and (data_config.train.PSD == 'design')):
                
#                    f_test = h5py.File(data_config.data.NSBH.path_test_design_Bayestar_test, 'r')
                    
#                    h1_test_real = np.real(f_test['h1_snr_series'][1000:3500][()])
#                    l1_test_real = np.real(f_test['l1_snr_series'][1000:3500][()])
#                    v1_test_real = np.real(f_test['v1_snr_series'][1000:3500][()])
        
#                    h1_test_imag = np.imag((f_test['h1_snr_series'][1000:3500][()]))
#                    l1_test_imag = np.imag((f_test['l1_snr_series'][1000:3500][()]))
#                    v1_test_imag = np.imag((f_test['v1_snr_series'][1000:3500][()]))
                                      
#                    f_test.close() 
                    
                    f_test = h5py.File(data_config.data.NSBH.path_test_Bayestar_post_merger, 'r')
                    
                    h1_test_real = np.real(f_test['h1_snr_series'][()])
                    l1_test_real = np.real(f_test['l1_snr_series'][()])
                    v1_test_real = np.real(f_test['v1_snr_series'][()])
        
                    h1_test_imag = np.imag((f_test['h1_snr_series'][()]))
                    l1_test_imag = np.imag((f_test['l1_snr_series'][()]))
                    v1_test_imag = np.imag((f_test['v1_snr_series'][()]))
                                      
                    f_test.close() 
                
                
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                    
                    f_test = h5py.File(data_config.data.NSBH.path_test_low_snr, 'r')
                   
                    group_test = f_test['omf_injection_snr_samples']
        
                    data_h1_test = group_test['h1_snr']
                    data_l1_test = group_test['l1_snr']
                    data_v1_test = group_test['v1_snr']
        
                    h1_test_real = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    l1_test_real = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    v1_test_real = np.zeros([data_config.train.num_test, data_config.train.n_samples])
        
                    h1_test_imag = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    l1_test_imag = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    v1_test_imag = np.zeros([data_config.train.num_test, data_config.train.n_samples])
        
                    for i in range(data_config.train.num_test):
                
                        h1_test_real[i] = np.real(data_h1_test[str(i)][()][1840:2250] )
                        l1_test_real[i] = np.real(data_l1_test[str(i)][()][1840:2250] )
                        v1_test_real[i] = np.real(data_v1_test[str(i)][()][1840:2250] )
    
                        h1_test_imag[i] = np.realnp.imag((data_h1_test[str(i)][()][1840:2250]))
                        l1_test_imag[i] = np.realnp.imag((data_l1_test[str(i)][()][1840:2250]))
                        v1_test_imag[i] = np.realnp.imag((data_v1_test[str(i)][()][1840:2250]))
        
                    f_test.close()
            
            
            elif(data_config.train.dataset == 'BBH'):
                
                if((data_config.train.test_real == True) and (data_config.train.PSD == 'aLIGO')):
                    f1 = h5py.File(data_config.data.BBH.path_test_GW170729, 'r')
                    f2 = h5py.File(data_config.data.BBH.path_test_GW170809, 'r')
                    f3 = h5py.File(data_config.data.BBH.path_test_GW170814, 'r')
                    f4 = h5py.File(data_config.data.BBH.path_test_GW170818, 'r')
                    
                    h1_real_1 = np.real(f1['h1_snr_series'][()])
                    l1_real_1 = np.real(f1['l1_snr_series'][()])
                    v1_real_1 = np.real(f1['v1_snr_series'][()])
        
                    h1_real_2 = np.real(f2['h1_snr_series'][()])
                    l1_real_2 = np.real(f2['l1_snr_series'][()])
                    v1_real_2 = np.real(f2['v1_snr_series'][()])
                    
                    h1_real_3 = np.real(f3['h1_snr_series'][()])
                    l1_real_3 = np.real(f3['l1_snr_series'][()])
                    v1_real_3 = np.real(f3['v1_snr_series'][()])
        
                    h1_real_4 = np.real(f4['h1_snr_series'][()])
                    l1_real_4 = np.real(f4['l1_snr_series'][()])
                    v1_real_4 = np.real(f4['v1_snr_series'][()])
                    
                    
                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
        
                    h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
        
                    h1_imag_4 = np.imag((f4['h1_snr_series'][()]))
                    l1_imag_4 = np.imag((f4['l1_snr_series'][()]))
                    v1_imag_4 = np.imag((f4['v1_snr_series'][()]))
                    
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
                    
                
                if((data_config.train.test_real == True) and (data_config.train.PSD == 'O3')):
                    
                    f1 = h5py.File(data_config.data.BBH.path_test_GW190521, 'r')
                    f2 = h5py.File(data_config.data.BBH.path_test_GW190412, 'r')
                    f3 = h5py.File(data_config.data.BBH.path_test_GW200224_22223, 'r')
                    
                    h1_real_1 = np.real(f1['h1_snr_series'][()])
                    l1_real_1 = np.real(f1['l1_snr_series'][()])
                    v1_real_1 = np.real(f1['v1_snr_series'][()])
                    
                    h1_imag_1 = np.imag((f1['h1_snr_series'][()]))
                    l1_imag_1 = np.imag((f1['l1_snr_series'][()]))
                    v1_imag_1 = np.imag((f1['v1_snr_series'][()]))
                    
                    h1_real_2 = np.real(f2['h1_snr_series'][()])
                    l1_real_2 = np.real(f2['l1_snr_series'][()])
                    v1_real_2 = np.real(f2['v1_snr_series'][()])
                    
                    h1_imag_2 = np.imag((f2['h1_snr_series'][()]))
                    l1_imag_2 = np.imag((f2['l1_snr_series'][()]))
                    v1_imag_2 = np.imag((f2['v1_snr_series'][()]))
                    
                    h1_real_3 = np.real(f3['h1_snr_series'][()])
                    l1_real_3 = np.real(f3['l1_snr_series'][()])
                    v1_real_3 = np.real(f3['v1_snr_series'][()])
                    
                    h1_imag_3 = np.imag((f3['h1_snr_series'][()]))
                    l1_imag_3 = np.imag((f3['l1_snr_series'][()]))
                    v1_imag_3 = np.imag((f3['v1_snr_series'][()]))
        
                    h1_real_1 = h1_real_1[None,:]
                    l1_real_1 = l1_real_1[None,:]                    
                    v1_real_1 = v1_real_1[None,:]                    

                    h1_imag_1 = h1_imag_1[None,:]                   
                    l1_imag_1 = l1_imag_1[None,:]                    
                    v1_imag_1 = v1_imag_1[None,:] 
                    
                    h1_real_2 = h1_real_2[None,:]
                    l1_real_2 = l1_real_2[None,:]                    
                    v1_real_2 = v1_real_2[None,:]                    

                    h1_imag_2 = h1_imag_2[None,:]                   
                    l1_imag_2 = l1_imag_2[None,:]                    
                    v1_imag_2 = v1_imag_2[None,:] 
                    
                    h1_real_3 = h1_real_3[None,:]
                    l1_real_3 = l1_real_3[None,:]                    
                    v1_real_3 = v1_real_3[None,:]                    

                    h1_imag_3 = h1_imag_3[None,:]                   
                    l1_imag_3 = l1_imag_3[None,:]                    
                    v1_imag_3 = v1_imag_3[None,:] 
                    
           
                    h1_test_real = np.concatenate([h1_real_1, h1_real_2, h1_real_3], axis=0)
                    l1_test_real = np.concatenate([l1_real_1, l1_real_2, l1_real_3], axis=0)
                    v1_test_real = np.concatenate([v1_real_1, v1_real_2, v1_real_3], axis=0)
            
                    h1_test_imag = np.concatenate([h1_imag_1, h1_imag_2, h1_imag_3], axis=0)
                    l1_test_imag = np.concatenate([l1_imag_1, l1_imag_2, l1_imag_3], axis=0)
                    v1_test_imag = np.concatenate([v1_imag_1, v1_imag_2, v1_imag_3], axis=0)
                    
                    f1.close()
                    f2.close()
                    f3.close()

                
                    
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                    
                    f1 = h5py.File(data_config.data.BBH.path_test_low_SNR, 'r')

                    h1_test_real = np.real(f1['h1_snr_series'][0:2000][()])
                    l1_test_real = np.real(f1['l1_snr_series'][0:2000][()])
                    v1_test_real = np.real(f1['v1_snr_series'][0:2000][()])
       
                    h1_test_imag = np.imag((f1['h1_snr_series'][0:2000][()]))
                    l1_test_imag = np.imag((f1['l1_snr_series'][0:2000][()]))
                    v1_test_imag = np.imag((f1['v1_snr_series'][0:2000][()]))
            
                    f1.close()
                
                elif((data_config.train.test_real == False) and (data_config.train.PSD == 'design')):
                    

#                    f_test = h5py.File(data_config.data.BBH.path_test_design_Bayestar_test, 'r')
                    f_test = h5py.File(data_config.data.BBH.path_test_Bayestar_post_merger, 'r')
                   
                    h1_test_real = np.real(f_test['h1_snr_series'][()])
                    l1_test_real = np.real(f_test['l1_snr_series'][()])
                    v1_test_real = np.real(f_test['v1_snr_series'][()])
        
                    h1_test_imag = np.imag((f_test['h1_snr_series'][()]))
                    l1_test_imag = np.imag((f_test['l1_snr_series'][()]))
                    v1_test_imag = np.imag((f_test['v1_snr_series'][()]))
                                      
                    f_test.close()
                    
            elif(data_config.train.dataset == 'BNS'):
                    
                if((data_config.train.test_real == True) and (data_config.train.PSD == 'O2')):
            
                    f_test_GW170817 = h5py.File(data_config.data.BNS.path_test_GW170817, 'r')
                
                    h1_test_real_GW170817 = np.real(f_test_GW170817['h1_snr_series'][()])
                    l1_test_real_GW170817 = np.real(f_test_GW170817['l1_snr_series'][()])
                    v1_test_real_GW170817 = np.real(f_test_GW170817['v1_snr_series'][()])
        
                    h1_test_imag_GW170817 = np.imag((f_test_GW170817['h1_snr_series'][()]))
                    l1_test_imag_GW170817 = np.imag((f_test_GW170817['l1_snr_series'][()]))
                    v1_test_imag_GW170817 = np.imag((f_test_GW170817['v1_snr_series'][()]))  
                
                    h1_test_real_GW170817 = h1_test_real_GW170817[None,:]
                    l1_test_real_GW170817 = l1_test_real_GW170817[None,:]
                    v1_test_real_GW170817 = v1_test_real_GW170817[None,:]
                
                    h1_test_imag_GW170817 = h1_test_imag_GW170817[None,:]
                    l1_test_imag_GW170817 = l1_test_imag_GW170817[None,:]
                    v1_test_imag_GW170817 = v1_test_imag_GW170817[None,:]
                
                
                    h1_test_real = h1_test_real_GW170817
                    l1_test_real = l1_test_real_GW170817
                    v1_test_real = v1_test_real_GW170817
                
                    h1_test_imag = h1_test_imag_GW170817
                    l1_test_imag = l1_test_imag_GW170817
                    v1_test_imag = v1_test_imag_GW170817
                
                    f_test_GW170817.close() 
                
                            
                
                elif((data_config.train.test_real == False) and (data_config.train.PSD == 'design')):
                    
    
#                    f_test = h5py.File(data_config.data.BNS.path_test_design_Bayestar_test, 'r')
                    f_test = h5py.File(data_config.data.BNS.path_test_Bayestar_post_merger, 'r')

                    h1_test_real = np.real(f_test['h1_snr_series'][0:10][()])
                    l1_test_real = np.real(f_test['l1_snr_series'][0:10][()])
                    v1_test_real = np.real(f_test['v1_snr_series'][0:10][()])
        
                    h1_test_imag = np.imag((f_test['h1_snr_series'][0:10][()]))
                    l1_test_imag = np.imag((f_test['l1_snr_series'][0:10][()]))
                    v1_test_imag = np.imag((f_test['v1_snr_series'][0:10][()]))
            
                    f_test.close()

    
                
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                    
                    f_test = h5py.File(data_config.data.BNS.path_test_low_SNR, 'r')
                    
                    group_test = f_test['omf_injection_snr_samples']
        
                    data_h1_test = group_test['h1_snr']
                    data_l1_test = group_test['l1_snr']
                    data_v1_test = group_test['v1_snr']
        
                    h1_test_real = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    l1_test_real = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    v1_test_real = np.zeros([data_config.train.num_test, data_config.train.n_samples])
        
                    h1_test_imag = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    l1_test_imag = np.zeros([data_config.train.num_test, data_config.train.n_samples])
                    v1_test_imag = np.zeros([data_config.train.num_test, data_config.train.n_samples])
        
                    for i in range(data_config.train.num_test):
                
                        h1_test_real[i] = np.real(data_h1_test[str(i)][()][1840:2250] )
                        l1_test_real[i] = np.real(data_l1_test[str(i)][()][1840:2250] )
                        v1_test_real[i] = np.real(data_v1_test[str(i)][()][1840:2250] )
    
                        h1_test_imag[i] = np.realnp.imag((data_h1_test[str(i)][()][1840:2250]))
                        l1_test_imag[i] = np.realnp.imag((data_l1_test[str(i)][()][1840:2250]))
                        v1_test_imag[i] = np.realnp.imag((data_v1_test[str(i)][()][1840:2250]))
        
                    f_test.close()
        
        elif(data_config.train.test_negative_latency == True):
            
            
            if((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'aLIGO') and (data_config.train.test_negative_latency_seconds=='0')):
                
                f1 = h5py.File(data_config.data.BNS.path_test_3_det_0_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][()])
                l1_test_real = np.real(f1['l1_snr_series'][()])
                v1_test_real = np.real(f1['v1_snr_series'][()])
        
                h1_test_imag = np.realnp.imag((f1['h1_snr_series'][()]))
                l1_test_imag = np.realnp.imag((f1['l1_snr_series'][()]))
                v1_test_imag = np.realnp.imag((f1['v1_snr_series'][()]))
                        
                f1.close()
                
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'aLIGO') and (data_config.train.test_negative_latency_seconds=='5')):
                
                f1 = h5py.File(data_config.data.BNS.path_test_3_det_5_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][()])
                l1_test_real = np.real(f1['l1_snr_series'][()])
                v1_test_real = np.real(f1['v1_snr_series'][()])
        
                h1_test_imag = np.imag((f1['h1_snr_series'][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][()]))
                        
                f1.close()
                
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'aLIGO') and (data_config.train.test_negative_latency_seconds=='10')):
                
                f1 = h5py.File(data_config.data.BNS.path_test_3_det_10_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][()])
                l1_test_real = np.real(f1['l1_snr_series'][()])
                v1_test_real = np.real(f1['v1_snr_series'][()])
        
                h1_test_imag = np.imag((f1['h1_snr_series'][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][()]))
                        
                f1.close()
                
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'aLIGO') and (data_config.train.test_negative_latency_seconds=='15')):
                
                f1 = h5py.File(data_config.data.BNS.path_test_3_det_15_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][()])
                l1_test_real = np.real(f1['l1_snr_series'][()])
                v1_test_real = np.real(f1['v1_snr_series'][()])
        
                h1_test_imag = np.imag((f1['h1_snr_series'][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][()]))
                        
                f1.close()
                
                    
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'design') and (data_config.train.test_negative_latency_seconds=='0')):
                
                
                f_test = h5py.File(data_config.data.BNS.path_test_design, 'r')
                f_test_GW170817 = h5py.File(data_config.data.BNS.path_test_GW170817, 'r')
                   
                h1_test_real = np.real(f_test['h1_snr_series'][()])
                l1_test_real = np.real(f_test['l1_snr_series'][()])
                v1_test_real = np.real(f_test['v1_snr_series'][()])
        
                h1_test_imag = np.imag((f_test['h1_snr_series'][()]))
                l1_test_imag = np.imag((f_test['l1_snr_series'][()]))
                v1_test_imag = np.imag((f_test['v1_snr_series'][()]))
                
                h1_test_real_GW170817 = np.real(f_test_GW170817['h1_snr_series'][()])
                l1_test_real_GW170817 = np.real(f_test_GW170817['l1_snr_series'][()])
                v1_test_real_GW170817 = np.real(f_test_GW170817['v1_snr_series'][()])
        
                h1_test_imag_GW170817 = np.imag((f_test_GW170817['h1_snr_series'][()]))
                l1_test_imag_GW170817 = np.imag((f_test_GW170817['l1_snr_series'][()]))
                v1_test_imag_GW170817 = np.imag((f_test_GW170817['v1_snr_series'][()]))  
                
                h1_test_real_GW170817 = h1_test_real_GW170817[None,:]
                l1_test_real_GW170817 = l1_test_real_GW170817[None,:]
                v1_test_real_GW170817 = v1_test_real_GW170817[None,:]
                
                h1_test_imag_GW170817 = h1_test_imag_GW170817[None,:]
                l1_test_imag_GW170817 = l1_test_imag_GW170817[None,:]
                v1_test_imag_GW170817 = v1_test_imag_GW170817[None,:]
                
                
                h1_test_real = np.concatenate([h1_test_real, h1_test_real_GW170817], axis=0)
                l1_test_real = np.concatenate([l1_test_real, l1_test_real_GW170817], axis=0)
                v1_test_real = np.concatenate([v1_test_real, v1_test_real_GW170817], axis=0)
                
                h1_test_imag = np.concatenate([h1_test_imag, h1_test_imag_GW170817], axis=0)
                l1_test_imag = np.concatenate([l1_test_imag, l1_test_imag_GW170817], axis=0)
                v1_test_imag = np.concatenate([v1_test_imag, v1_test_imag_GW170817], axis=0)
                
                
                h1_test_real = np.array([h1_test_real[-1]])
                l1_test_real = np.array([l1_test_real[-1]])
                v1_test_real = np.array([v1_test_real[-1]])
                
                h1_test_imag = np.array([h1_test_imag[-1]])
                l1_test_imag = np.array([l1_test_imag[-1]])
                v1_test_imag = np.array([v1_test_imag[-1]])
                
                                      
                f_test.close()
                f_test_GW170817.close()
                      
                
                    
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'design') and (data_config.train.test_negative_latency_seconds=='10')):
                

                f1 = h5py.File(data_config.data.BNS.path_test_design_10_secs, 'r')
                f2 = h5py.File(data_config.data.BNS.path_test_GW170817_10_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][0:2500][()]) # 35000
                l1_test_real = np.real(f1['l1_snr_series'][0:2500][()])
                v1_test_real = np.real(f1['v1_snr_series'][0:2500][()])
        
                h1_test_imag = np.imag((f1['h1_snr_series'][0:2500][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][0:2500][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][0:2500][()]))                
            
                h1_test_real_GW170817 = np.real(f2['h1_snr_series'][()]) # 35000
                l1_test_real_GW170817 = np.real(f2['l1_snr_series'][()])
                v1_test_real_GW170817 = np.real(f2['v1_snr_series'][()])
        
                h1_test_imag_GW170817 = np.imag((f2['h1_snr_series'][()]))
                l1_test_imag_GW170817 = np.imag((f2['l1_snr_series'][()]))
                v1_test_imag_GW170817 = np.imag((f2['v1_snr_series'][()]))       
                
                h1_test_real = np.concatenate([h1_test_real, h1_test_real_GW170817], axis=0)
                l1_test_real = np.concatenate([l1_test_real, l1_test_real_GW170817], axis=0)
                v1_test_real = np.concatenate([v1_test_real, v1_test_real_GW170817], axis=0)
                
                h1_test_imag = np.concatenate([h1_test_imag, h1_test_imag_GW170817], axis=0)
                l1_test_imag = np.concatenate([l1_test_imag, l1_test_imag_GW170817], axis=0)
                v1_test_imag = np.concatenate([v1_test_imag, v1_test_imag_GW170817], axis=0)
                        
                f1.close()
                f2.close()
                
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'design') and (data_config.train.test_negative_latency_seconds=='15')):
                

                f1 = h5py.File(data_config.data.BNS.path_test_design_15_secs, 'r')
                f2 = h5py.File(data_config.data.BNS.path_test_GW170817_15_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][0:3000][()]) # 28000
                l1_test_real = np.real(f1['l1_snr_series'][0:3000][()])
                v1_test_real = np.real(f1['v1_snr_series'][0:3000][()])
        
                h1_test_imag = np.imag((f1['h1_snr_series'][0:3000][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][0:3000][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][0:3000][()]))
                
                h1_test_real_GW170817 = np.real(f2['h1_snr_series'][()]) # 35000
                l1_test_real_GW170817 = np.real(f2['l1_snr_series'][()])
                v1_test_real_GW170817 = np.real(f2['v1_snr_series'][()])
        
                h1_test_imag_GW170817 = np.imag((f2['h1_snr_series'][()]))
                l1_test_imag_GW170817 = np.imag((f2['l1_snr_series'][()]))
                v1_test_imag_GW170817 = np.imag((f2['v1_snr_series'][()]))        
                
                h1_test_real = np.concatenate([h1_test_real, h1_test_real_GW170817], axis=0)
                l1_test_real = np.concatenate([l1_test_real, l1_test_real_GW170817], axis=0)
                v1_test_real = np.concatenate([v1_test_real, v1_test_real_GW170817], axis=0)
                
                h1_test_imag = np.concatenate([h1_test_imag, h1_test_imag_GW170817], axis=0)
                l1_test_imag = np.concatenate([l1_test_imag, l1_test_imag_GW170817], axis=0)
                v1_test_imag = np.concatenate([v1_test_imag, v1_test_imag_GW170817], axis=0)
                        
                f1.close()
                f2.close()
                
                
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'design') and (data_config.train.test_negative_latency_seconds=='30')):
                
                f1 = h5py.File(data_config.data.BNS.path_test_design_30_secs, 'r')
                f2 = h5py.File(data_config.data.BNS.path_test_GW170817_30_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][0:3000][()]) # 12000
                l1_test_real = np.real(f1['l1_snr_series'][0:3000][()])
                v1_test_real = np.real(f1['v1_snr_series'][0:3000][()])
       
                h1_test_imag = np.imag((f1['h1_snr_series'][0:3000][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][0:3000][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][0:3000][()]))
                
                h1_test_real_GW170817 = np.real(f2['h1_snr_series'][()]) # 35000
                l1_test_real_GW170817 = np.real(f2['l1_snr_series'][()])
                v1_test_real_GW170817 = np.real(f2['v1_snr_series'][()])
        
                h1_test_imag_GW170817 = np.imag((f2['h1_snr_series'][()]))
                l1_test_imag_GW170817 = np.imag((f2['l1_snr_series'][()]))
                v1_test_imag_GW170817 = np.imag((f2['v1_snr_series'][()])) 
                
                h1_test_real = np.concatenate([h1_test_real, h1_test_real_GW170817], axis=0)
                l1_test_real = np.concatenate([l1_test_real, l1_test_real_GW170817], axis=0)
                v1_test_real = np.concatenate([v1_test_real, v1_test_real_GW170817], axis=0)
                
                h1_test_imag = np.concatenate([h1_test_imag, h1_test_imag_GW170817], axis=0)
                l1_test_imag = np.concatenate([l1_test_imag, l1_test_imag_GW170817], axis=0)
                v1_test_imag = np.concatenate([v1_test_imag, v1_test_imag_GW170817], axis=0)                       
                        
                f1.close()
                f2.close()
                
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'design') and (data_config.train.test_negative_latency_seconds=='45')):
                
                f1 = h5py.File(data_config.data.BNS.path_test_design_45_secs, 'r')
                f2 = h5py.File(data_config.data.BNS.path_test_GW170817_45_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][0:3000][()]) # 8000
                l1_test_real = np.real(f1['l1_snr_series'][0:3000][()])
                v1_test_real = np.real(f1['v1_snr_series'][0:3000][()])
        
                h1_test_imag = np.imag((f1['h1_snr_series'][0:3000][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][0:3000][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][0:3000][()]))
                
                h1_test_real_GW170817 = np.real(f2['h1_snr_series'][()]) # 35000
                l1_test_real_GW170817 = np.real(f2['l1_snr_series'][()])
                v1_test_real_GW170817 = np.real(f2['v1_snr_series'][()])
        
                h1_test_imag_GW170817 = np.imag((f2['h1_snr_series'][()]))
                l1_test_imag_GW170817 = np.imag((f2['l1_snr_series'][()]))
                v1_test_imag_GW170817 = np.imag((f2['v1_snr_series'][()]))          
                
                h1_test_real = np.concatenate([h1_test_real, h1_test_real_GW170817], axis=0)
                l1_test_real = np.concatenate([l1_test_real, l1_test_real_GW170817], axis=0)
                v1_test_real = np.concatenate([v1_test_real, v1_test_real_GW170817], axis=0)
                
                h1_test_imag = np.concatenate([h1_test_imag, h1_test_imag_GW170817], axis=0)
                l1_test_imag = np.concatenate([l1_test_imag, l1_test_imag_GW170817], axis=0)
                v1_test_imag = np.concatenate([v1_test_imag, v1_test_imag_GW170817], axis=0)                       
                        
                f1.close()
                f2.close()
                
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'design') and (data_config.train.test_negative_latency_seconds=='58')):
                
                f1 = h5py.File(data_config.data.BNS.path_test_design_58_secs, 'r')
                f2 = h5py.File(data_config.data.BNS.path_test_GW170817_58_secs, 'r')
            
                h1_test_real = np.real(f1['h1_snr_series'][0:3000][()]) # 7000
                l1_test_real = np.real(f1['l1_snr_series'][0:3000][()])
                v1_test_real = np.real(f1['v1_snr_series'][0:3000][()])
        
                h1_test_imag = np.imag((f1['h1_snr_series'][0:3000][()]))
                l1_test_imag = np.imag((f1['l1_snr_series'][0:3000][()]))
                v1_test_imag = np.imag((f1['v1_snr_series'][0:3000][()]))
                
                h1_test_real_GW170817 = np.real(f2['h1_snr_series'][()]) # 35000
                l1_test_real_GW170817 = np.real(f2['l1_snr_series'][()])
                v1_test_real_GW170817 = np.real(f2['v1_snr_series'][()])
        
                h1_test_imag_GW170817 = np.imag((f2['h1_snr_series'][()]))
                l1_test_imag_GW170817 = np.imag((f2['l1_snr_series'][()]))
                v1_test_imag_GW170817 = np.imag((f2['v1_snr_series'][()]))         
                
                h1_test_real = np.concatenate([h1_test_real, h1_test_real_GW170817], axis=0)
                l1_test_real = np.concatenate([l1_test_real, l1_test_real_GW170817], axis=0)
                v1_test_real = np.concatenate([v1_test_real, v1_test_real_GW170817], axis=0)
                
                h1_test_imag = np.concatenate([h1_test_imag, h1_test_imag_GW170817], axis=0)
                l1_test_imag = np.concatenate([l1_test_imag, l1_test_imag_GW170817], axis=0)
                v1_test_imag = np.concatenate([v1_test_imag, v1_test_imag_GW170817], axis=0)        
                        
                f1.close()
                f2.close()
                
            
        
            elif((data_config.train.dataset == 'BNS') and (psd == 'O4')):
                
                if(test_negative_latency_seconds == '0'):
                    
                    f_test = h5py.File(data_config.data.BNS.path_test_O4_PSD_0_sec, 'r')
                    
                    h1_test_real = np.real(f_test['h1_snr_series'][()])
                    l1_test_real = np.real(f_test['l1_snr_series'][()])
                    v1_test_real = np.real(f_test['v1_snr_series'][()])
        
                    h1_test_imag = np.realnp.imag((f_test['h1_snr_series'][()]))
                    l1_test_imag = np.realnp.imag((f_test['l1_snr_series'][()]))
                    v1_test_imag = np.realnp.imag((f_test['v1_snr_series'][()]))
                    
                    f_test.close()
                
                elif(test_negative_latency_seconds == '5'):
                    
                    f_test = h5py.File(data_config.data.BNS.path_test_O4_PSD_5_sec, 'r')
                    
                    h1_test_real = np.real(f_test['h1_snr_series'][()])
                    l1_test_real = np.real(f_test['l1_snr_series'][()])
                    v1_test_real = np.real(f_test['v1_snr_series'][()])
        
                    h1_test_imag = np.realnp.imag((f_test['h1_snr_series'][()]))
                    l1_test_imag = np.realnp.imag((f_test['l1_snr_series'][()]))
                    v1_test_imag = np.realnp.imag((f_test['v1_snr_series'][()]))
                    
                    f_test.close()
                    
                elif(test_negative_latency_seconds == '10'):
                    
                    f_test = h5py.File(data_config.data.BNS.path_test_O4_PSD_10_sec, 'r')
                    
                    h1_test_real = np.real(f_test['h1_snr_series'][()])
                    l1_test_real = np.real(f_test['l1_snr_series'][()])
                    v1_test_real = np.real(f_test['v1_snr_series'][()])
        
                    h1_test_imag = np.realnp.imag((f_test['h1_snr_series'][()]))
                    l1_test_imag = np.realnp.imag((f_test['l1_snr_series'][()]))
                    v1_test_imag = np.realnp.imag((f_test['v1_snr_series'][()]))
                    
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
    def load_test_2_det_data(data_config):
        """Loads dataset from path"""
        #Get the HDF5 group
        #BNS
        if((data_config.train.dataset == 'BNS') and (data_config.train.snr_range_test == 'low') and (data_config.train.test_negative_latency == False)):
            f1 = h5py.File(data_config.data.BNS.path_test_2_det_low_SNR, 'r')
            f2 = h5py.File(data_config.data.BNS.path_test, 'r')
            f3 = h5py.File(data_config.data.BNS.path_test_2_det_high_SNR, 'r')

            h1_test_real_1 = np.real(f1['h1_snr_series'][()])
            l1_test_real_1 = np.real(f1['l1_snr_series'][()])
        
            h1_test_imag_1 = np.imag((f1['h1_snr_series'][()]))
            l1_test_imag_1 = np.imag((f1['l1_snr_series'][()]))
                        
            h1_test_real_2 = np.real(f2['h1_snr_series'][()])
            l1_test_real_2 = np.real(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_test_imag_2 = np.imag((f2['l1_snr_series'][()]))
            
            h1_test_real_3 = np.real(f3['h1_snr_series'][()])
            l1_test_real_3 = np.real(f3['l1_snr_series'][()])
        
            h1_test_imag_3 = np.imag((f3['h1_snr_series'][()]))
            l1_test_imag_3 = np.imag((f3['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_1, h1_test_real_2, h1_test_real_3], axis=0)
            l1_test_real = np.concatenate([l1_test_real_1, l1_test_real_2, l1_test_real_3], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_1, h1_test_imag_2, h1_test_imag_3], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_1, l1_test_imag_2, l1_test_imag_3], axis=0)
            
            f1.close()
            f2.close()
            f3.close()
            
        if((data_config.train.dataset == 'BNS') and (data_config.train.test_negative_latency == True)):
                 
#            f1 = h5py.File(data_config.data.BNS.path_test_2_det_0_secs, 'r')
            f2 = h5py.File(data_config.data.BNS.path_test_2_det_5_secs, 'r')
            f3 = h5py.File(data_config.data.BNS.path_test_2_det_10_secs, 'r')
            f4 = h5py.File(data_config.data.BNS.path_test_2_det_15_secs, 'r')
            
#            h1_test_real_1 = np.real(f1['h1_snr_series'][()])
#            l1_test_real_1 = np.real(f1['l1_snr_series'][()])
        
#            h1_test_imag_1 = np.realnp.imag((f1['h1_snr_series'][()]))
#            l1_test_imag_1 = np.realnp.imag((f1['l1_snr_series'][()]))
                        
            h1_test_real_2 = np.real(f2['h1_snr_series'][()])
            l1_test_real_2 = np.real(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_test_imag_2 = np.imag((f2['l1_snr_series'][()]))
            
            h1_test_real_3 = np.real(f3['h1_snr_series'][()])
            l1_test_real_3 = np.real(f3['l1_snr_series'][()])
        
            h1_test_imag_3 = np.imag((f3['h1_snr_series'][()]))
            l1_test_imag_3 = np.imag((f3['l1_snr_series'][()]))
                        
            h1_test_real_4 = np.real(f4['h1_snr_series'][()])
            l1_test_real_4 = np.real(f4['l1_snr_series'][()])
        
            h1_test_imag_4 = np.imag((f4['h1_snr_series'][()]))
            l1_test_imag_4 = np.imag((f4['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_2, h1_test_real_3, h1_test_real_4], axis=0)
            l1_test_real = np.concatenate([l1_test_real_2, l1_test_real_3, l1_test_real_4], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_2, h1_test_imag_3, h1_test_imag_4], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_2, l1_test_imag_3, l1_test_imag_4], axis=0)
            
#            f1.close()
            f2.close()
            f3.close()
            f4.close()
            
            
        #NSBH
        elif((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_test == 'low')):
                 
            f1 = h5py.File(data_config.data.NSBH.path_test, 'r')
            f2 = h5py.File(data_config.data.NSBH.path_test_low_snr, 'r')

            h1_test_real_1 = np.real(f1['h1_snr_series'][()])
            l1_test_real_1 = np.real(f1['l1_snr_series'][()])
        
            h1_test_imag_1 = np.imag((f1['h1_snr_series'][()]))
            l1_test_imag_1 = np.imag((f1['l1_snr_series'][()]))
                        
            h1_test_real_2 = np.real(f2['h1_snr_series'][()])
            l1_test_real_2 = np.real(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_test_imag_2 = np.imag((f2['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_1, h1_test_real_2], axis=0)
            l1_test_real = np.concatenate([l1_test_real_1, l1_test_real_2], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_1, h1_test_imag_2], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_1, l1_test_imag_2], axis=0)
            
            f1.close()
            f2.close()
            
        #BBH
        elif((data_config.train.dataset == 'BBH') and (data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low')):
                 
            f1 = h5py.File(data_config.data.BBH.path_train, 'r')
            f2 = h5py.File(data_config.data.BBH.path_test_low_SNR, 'r')

            h1_test_real_1 = np.real(f1['h1_snr_series'][0:2000][()])
            l1_test_real_1 = np.real(f1['l1_snr_series'][0:2000][()])
       
            h1_test_imag_1 = np.imag((f1['h1_snr_series'][0:2000][()]))
            l1_test_imag_1 = np.imag((f1['l1_snr_series'][0:2000][()]))
                        
            h1_test_real_2 = np.real(f2['h1_snr_series'][()])
            l1_test_real_2 = np.real(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_test_imag_2 = np.imag((f2['l1_snr_series'][()]))
            
            h1_test_real = np.concatenate([h1_test_real_1, h1_test_real_2], axis=0)
            l1_test_real = np.concatenate([l1_test_real_1, l1_test_real_2], axis=0)
            
            h1_test_imag = np.concatenate([h1_test_imag_1, h1_test_imag_2], axis=0)
            l1_test_imag = np.concatenate([l1_test_imag_1, l1_test_imag_2], axis=0)
            
            f1.close()
            f2.close()
        
        
        elif((data_config.train.dataset == 'BBH') and (data_config.train.test_real == True) and (data_config.train.PSD == 'O2')):
            
            f1 = h5py.File(data_config.data.BBH.path_test_GW150914, 'r')
            f2 = h5py.File(data_config.data.BBH.path_test_GW170104, 'r')

            h1_test_real_1 = np.real(f1['h1_snr_series'][()])
            l1_test_real_1 = np.real(f1['l1_snr_series'][()])
       
            h1_test_imag_1 = np.imag((f1['h1_snr_series'][()]))
            l1_test_imag_1 = np.imag((f1['l1_snr_series'][()]))
            
            h1_test_real_1 = h1_test_real_1[None,:]
            l1_test_real_1 = l1_test_real_1[None,:]
                
            h1_test_imag_1 = h1_test_imag_1[None,:]
            l1_test_imag_1 = l1_test_imag_1[None,:]
                                    
            h1_test_real_2 = np.real(f2['h1_snr_series'][()])
            l1_test_real_2 = np.real(f2['l1_snr_series'][()])
        
            h1_test_imag_2 = np.imag((f2['h1_snr_series'][()]))
            l1_test_imag_2 = np.imag((f2['l1_snr_series'][()]))
            
            h1_test_real_2 = h1_test_real_2[None,:]
            l1_test_real_2 = l1_test_real_2[None,:]
                
            h1_test_imag_2 = h1_test_imag_2[None,:]
            l1_test_imag_2 = l1_test_imag_2[None,:]
            
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
                           
    @staticmethod
    def load_test_3_det_parameters(data_config):
        """Loads train parameters from path"""
                 
        if(data_config.train.test_negative_latency == False):
            if(data_config.train.dataset == 'NSBH'):
                if((data_config.train.snr_range_test == 'high') and (data_config.train.psd == 'aLIGO')):
                    f_test = h5py.File(data_config.parameters.NSBH.path_test, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                    
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                    f_test = h5py.File(data_config.parameters.NSBH.path_test_low_snr, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                
                elif((data_config.train.test_real == True) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                    
                    f_test = h5py.File(data_config.parameters.NSBH.path_test_GW190917, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                
                elif((data_config.train.test_real == True) and (data_config.train.PSD == 'O2')):
                    
                    f_test = h5py.File(data_config.parameters.NSBH.path_test_GW190814, 'r')
#                    f_test = h5py.File(data_config.BBH.path_test_GW190412, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)
            
                    mass_1 = f_test['mass2'][()]
                    mass_2 = f_test['mass1'][()]
                    spin_1 = f_test['spin1z'][()]
                    spin_2 = f_test['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][()]
                    gps_time = f_test['gps_time'][()]

                    f_test.close()
                
                
                elif((data_config.train.test_real == False) and (data_config.train.PSD == 'design')):
        
                    # For CBC-SkyNet:
        
#                    f_test = h5py.File(data_config.parameters.NSBH.path_test_design_Bayestar_test, 'r')
                    
#                    data_ra = f_test['ra'][1000:3500][()]
#                    data_dec = f_test['dec'][1000:3500][()]
        
#                    ra_test = 2.0*np.pi*data_ra
#                    ra_test = ra_test - np.pi
#                    ra_test_x = np.cos(ra_test)
#                    ra_test_y = np.sin(ra_test)
        
#                    dec_test = np.arcsin(1.0 - 2.0*data_dec)
            
#                    mass_1 = f_test['mass2'][1000:3500][()]
#                    mass_2 = f_test['mass1'][1000:3500][()]
#                    spin_1 = f_test['spin1z'][1000:3500][()]
#                    spin_2 = f_test['spin2z'][1000:3500][()]
##                    inc = f_test['inclination'][()]
#                    inj_snr = f_test['Injection_SNR'][1000:3500][()]
    
#                    gps_time = [1187008882.4]
#                    gps_time = np.repeat(gps_time, 2500)

#                    f_test.close()
                
                
                    f_test = h5py.File(data_config.parameters.NSBH.path_test_Bayestar_post_merger, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)
            
                    mass_1 = f_test['mass2'][()]
                    mass_2 = f_test['mass1'][()]
                    spin_1 = f_test['spin1z'][()]
                    spin_2 = f_test['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][()]
    
                    gps_time = f_test['gps_time'][()]
        
            
            elif(data_config.train.dataset == 'BBH'):
                if((data_config.train.test_real == True) and (data_config.train.PSD == 'aLIGO')):
                 
                    f1 = h5py.File(data_config.parameters.BBH.path_test_GW170729, 'r')
                    f2 = h5py.File(data_config.parameters.BBH.path_test_GW170809, 'r')
                    f3 = h5py.File(data_config.parameters.BBH.path_test_GW170814, 'r')
                    f4 = h5py.File(data_config.parameters.BBH.path_test_GW170818, 'r')
                    
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
                    
                elif((data_config.train.test_real == True) and (data_config.train.PSD == 'O3')):
                    
                    f_test_1 = h5py.File(data_config.parameters.BBH.path_test_GW190521, 'r')
                    f_test_2 = h5py.File(data_config.parameters.BBH.path_test_GW190412, 'r')
                    
                    data_ra_1 = f_test_1['ra'][()]
                    data_dec_1 = f_test_1['dec'][()]
                    
                    data_ra_2 = f_test_2['ra'][()]
                    data_dec_2 = f_test_2['dec'][()]
        
                    ra_test_1 = 2.0*np.pi*data_ra_1
                    ra_test_1 = ra_test_1 - np.pi
                
                    ra_test_2 = 2.0*np.pi*data_ra_2
                    ra_test_2 = ra_test_2 - np.pi
                    
                    dec_test_1 = np.arcsin(1.0 - 2.0*data_dec_1)
                    dec_test_2 = np.arcsin(1.0 - 2.0*data_dec_2)
                    
                    ra_test = np.concatenate([ra_test_1, ra_test_2])
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
            
                    dec_test = np.concatenate([dec_test_1, dec_test_2])  
                
                    mass_1_1 = f_test_1['mass2'][()]
                    mass_2_1 = f_test_1['mass1'][()]
                    spin_1_1 = f_test_1['spin1z'][()]
                    spin_2_1 = f_test_1['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_1 = f_test_1['Injection_SNR'][()]
                    gps_time_1 = f_test_1['gps_time'][()]
        
                    mass_1_2 = f_test_2['mass2'][()]
                    mass_2_2 = f_test_2['mass1'][()]
                    spin_1_2 = f_test_2['spin1z'][()]
                    spin_2_2 = f_test_2['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_2 = f_test_2['Injection_SNR'][()]
                    gps_time_2 = f_test_2['gps_time'][()]
                
                    ra_test = np.concatenate([ra_test_1, ra_test_2], axis=0)
                    dec_test = np.concatenate([dec_test_1, dec_test_2], axis=0)
                    mass_1 = np.concatenate([mass_1_1, mass_1_2], axis=0)
                    mass_2 = np.concatenate([mass_2_1, mass_2_2], axis=0)
                    spin_1 = np.concatenate([spin_1_1, spin_1_2], axis=0)
                    spin_2 = np.concatenate([spin_2_1, spin_2_2], axis=0)
                    inj_snr = np.concatenate([inj_snr_1, inj_snr_2], axis=0)       
                    gps_time = np.concatenate([gps_time_1, gps_time_2], axis=0)                    

                    f_test_1.close()
                    f_test_2.close()
        
        
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'high') and (data_config.train.PSD == 'aLIGO')):
                    
                    f_test = h5py.File(data_config.parameters.BBH.path_test, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                    
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                    
                    f_test = h5py.File(data_config.parameters.BBH.path_test_low_SNR, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
            
                
                elif((data_config.train.test_real == False) and (data_config.train.PSD == 'design')):
        
#                    f_test = h5py.File(data_config.parameters.BBH.path_test_design_Bayestar_test, 'r')
                    f_test = h5py.File(data_config.parameters.BBH.path_test_Bayestar_post_merger, 'r')
                   
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)
            
                    mass_1 = f_test['mass1'][()]
                    mass_2 = f_test['mass2'][()]
                    spin_1 = f_test['spin1z'][()]
                    spin_2 = f_test['spin2z'][()]
#                    inc = f_test['inclination'][()]
#                    inj_snr = f_test['Injection_SNR'][()]

#                    gps_time = [1187008882.4]
#                    gps_time = np.repeat(gps_time, len(ra_test))

                    gps_time = f_test['gps_time'][()]

                    f_test.close()
                
            
            
            elif(data_config.train.dataset == 'BNS'):
            
                
                if((data_config.train.test_real == True) and (data_config.train.PSD == 'O2')):
                    
                    f_test_GW170817 = h5py.File(data_config.parameters.BNS.path_test_GW170817_0_secs, 'r')
                     
                    ra_test = 2.0*np.pi*f_test_GW170817['ra'][()] # 35000
                    dec_test = np.arcsin(1.0-2.0*f_test_GW170817['dec'][()])
                    
                    mass_1 = f_test_GW170817['mass1'][()]
                    mass_2 = f_test_GW170817['mass2'][()]
                    spin_1 = f_test_GW170817['spin1z'][()]
                    spin_2 = f_test_GW170817['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test_GW170817['Injection_SNR'][()]
                    
                    gps_time = np.array([1187008882.4])
                    
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)                                        

                    f_test_GW170817.close()
                
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'high') and (data_config.train.PSD == 'aLIGO')):
                
                    f_test = h5py.File(data_config.parameters.BNS.path_test, 'r')
                    
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
                
                elif((data_config.train.test_real == False) and (data_config.train.PSD == 'design')):
                    
#                    f_test = h5py.File(data_config.parameters.BNS.path_test_design_Bayestar_test, 'r')
                    f_test = h5py.File(data_config.parameters.BNS.path_test_Bayestar_post_merger, 'r')
                    
                    data_ra = f_test['ra'][0:10][()] # 40000
                    data_dec = f_test['dec'][0:10][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)
            
                    mass_1 = f_test['mass1'][0:10][()]
                    mass_2 = f_test['mass2'][0:10][()]
                    spin_1 = f_test['spin1z'][0:10][()]
                    spin_2 = f_test['spin2z'][0:10][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][0:10][()]
                    gps_time = f_test['gps_time'][0:10][()]
    
#                    gps_time = [1187008882.4]
#                    gps_time = np.repeat(gps_time, len(ra_test))

                    f_test.close()                    
                
                elif((data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low') and (data_config.train.PSD == 'aLIGO')):
                
                    f_test = h5py.File(data_config.parameters.BNS.path_test_low_SNR, 'r')
                            
                    data_ra = f_test['ra'][()]
                    data_dec = f_test['dec'][()]
        
                    ra_test = 2.0*np.pi*data_ra
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
        
                    dec_test = np.arcsin(1.0 - 2.0*data_dec)

                    f_test.close()
        
        elif(data_config.train.test_negative_latency == True):
            
            if((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'aLIGO')):
                
                if(data_config.train.test_negative_latency_seconds == '5'):
              
                    f_test = h5py.File(data_config.parameters.BNS.path_test_3_det_5_secs, 'r')
           
                    ra_test = 2.0*np.pi*f_test['ra'][()]
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][()])
            
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)

                    f_test.close()
                    
                elif(data_config.train.test_negative_latency_seconds == '10'):
              
                    f_test = h5py.File(data_config.parameters.BNS.path_test_3_det_10_secs, 'r')
           
                    ra_test = 2.0*np.pi*f_test['ra'][()]
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][()])
            
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)

                    f_test.close()
                    
        
            
            elif((data_config.train.dataset == 'BNS') and ((data_config.train.PSD == 'design'))):
                
                if(data_config.train.test_negative_latency_seconds == '0'):
        
                    f_test = h5py.File(data_config.parameters.BNS.path_test_design, 'r')
                    f_test_GW170817 = h5py.File(data_config.parameters.BNS.path_test_GW170817_0_secs, 'r')
                    
                    ra_test = 2.0*np.pi*f_test['ra'][()] # 35000
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][()])
                    
                    ra_test_GW170817 = 2.0*np.pi*f_test_GW170817['ra'][()] # 35000
                    dec_test_GW170817 = np.arcsin(1.0-2.0*f_test_GW170817['dec'][()])
                    
                    mass_1 = f_test['mass1'][()]
                    mass_2 = f_test['mass2'][()]
                    spin_1 = f_test['spin1z'][()]
                    spin_2 = f_test['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][()]
    
                    mass_1_GW170817 = f_test_GW170817['mass1'][()]
                    mass_2_GW170817 = f_test_GW170817['mass2'][()]
                    spin_1_GW170817 = f_test_GW170817['spin1z'][()]
                    spin_2_GW170817 = f_test_GW170817['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_GW170817 = f_test_GW170817['Injection_SNR'][()]
    
                    ra_test = np.concatenate([ra_test, ra_test_GW170817], axis=0)
                    dec_test = np.concatenate([dec_test, dec_test_GW170817], axis=0)
                    mass_1 = np.concatenate([mass_1, mass_1_GW170817], axis=0)
                    mass_2 = np.concatenate([mass_2, mass_2_GW170817], axis=0)
                    spin_1 = np.concatenate([spin_1, spin_1_GW170817], axis=0)
                    spin_2 = np.concatenate([spin_2, spin_2_GW170817], axis=0)
                    inj_snr = np.concatenate([inj_snr, inj_snr_GW170817], axis=0)
                    
                    ra_test = np.array([ra_test[-1]])
                    dec_test = np.array([dec_test[-1]])
                    mass_1 = np.array([mass_1[-1]])
                    mass_2 = np.array([mass_2[-1]])
                    spin_1 = np.array([spin_1[-1]])
                    spin_2 = np.array([spin_2[-1]])
                    inj_snr = np.array([inj_snr[-1]])
                    
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)                                        

                    f_test.close()
                    f_test_GW170817.close()


                elif(data_config.train.test_negative_latency_seconds == '10'):


                    f_test = h5py.File(data_config.parameters.BNS.path_test_design_10_secs, 'r')
                    f_test_GW170817 = h5py.File(data_config.parameters.BNS.path_test_GW170817_10_secs, 'r')
           
                    ra_test = 2.0*np.pi*f_test['ra'][0:2500][()] # 35000
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][0:2500][()])
                    
                    ra_test_GW170817 = 2.0*np.pi*f_test_GW170817['ra'][()] # 35000
                    dec_test_GW170817 = np.arcsin(1.0-2.0*f_test_GW170817['dec'][()])
                    
                    mass_1 = f_test['mass1'][0:2500][()]
                    mass_2 = f_test['mass2'][0:2500][()]
                    spin_1 = f_test['spin1z'][0:2500][()]
                    spin_2 = f_test['spin2z'][0:2500][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][0:2500][()]
    
                    mass_1_GW170817 = f_test_GW170817['mass1'][()]
                    mass_2_GW170817 = f_test_GW170817['mass2'][()]
                    spin_1_GW170817 = f_test_GW170817['spin1z'][()]
                    spin_2_GW170817 = f_test_GW170817['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_GW170817 = f_test_GW170817['Injection_SNR'][()]
    
                    ra_test = np.concatenate([ra_test, ra_test_GW170817], axis=0)
                    dec_test = np.concatenate([dec_test, dec_test_GW170817], axis=0)
                    mass_1 = np.concatenate([mass_1, mass_1_GW170817], axis=0)
                    mass_2 = np.concatenate([mass_2, mass_2_GW170817], axis=0)
                    spin_1 = np.concatenate([spin_1, spin_1_GW170817], axis=0)
                    spin_2 = np.concatenate([spin_2, spin_2_GW170817], axis=0)
                    inj_snr = np.concatenate([inj_snr, inj_snr_GW170817], axis=0)  
                
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)                                        

                    f_test.close()
                    f_test_GW170817.close()

                    
                elif(data_config.train.test_negative_latency_seconds == '15'):


                    f_test = h5py.File(data_config.parameters.BNS.path_test_design_15_secs, 'r')
                    f_test_GW170817 = h5py.File(data_config.parameters.BNS.path_test_GW170817_15_secs, 'r')
           
                    ra_test = 2.0*np.pi*f_test['ra'][0:10][()] # 28000
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][0:10][()])
                    
                    ra_test_GW170817 = 2.0*np.pi*f_test_GW170817['ra'][()] # 35000
                    dec_test_GW170817 = np.arcsin(1.0-2.0*f_test_GW170817['dec'][()])
                                  
                
                    mass_1 = f_test['mass1'][0:10][()]
                    mass_2 = f_test['mass2'][0:10][()]
                    spin_1 = f_test['spin1z'][0:10][()]
                    spin_2 = f_test['spin2z'][0:10][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][0:10][()]
    
                    mass_1_GW170817 = f_test_GW170817['mass1'][()]
                    mass_2_GW170817 = f_test_GW170817['mass2'][()]
                    spin_1_GW170817 = f_test_GW170817['spin1z'][()]
                    spin_2_GW170817 = f_test_GW170817['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_GW170817 = f_test_GW170817['Injection_SNR'][()]
                    
                    ra_test = np.concatenate([ra_test, ra_test_GW170817], axis=0)
                    dec_test = np.concatenate([dec_test, dec_test_GW170817], axis=0)
                    mass_1 = np.concatenate([mass_1, mass_1_GW170817], axis=0)
                    mass_2 = np.concatenate([mass_2, mass_2_GW170817], axis=0)
                    spin_1 = np.concatenate([spin_1, spin_1_GW170817], axis=0)
                    spin_2 = np.concatenate([spin_2, spin_2_GW170817], axis=0)
                    inj_snr = np.concatenate([inj_snr, inj_snr_GW170817], axis=0)
            
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)

                    f_test.close()
                    f_test_GW170817.close()
                    
                elif(data_config.train.test_negative_latency_seconds == '30'):
              
                    f_test = h5py.File(data_config.parameters.BNS.path_test_design_30_secs, 'r')
                    f_test_GW170817 = h5py.File(data_config.parameters.BNS.path_test_GW170817_30_secs, 'r')
           
                    ra_test = 2.0*np.pi*f_test['ra'][0:3000][()] # 12000
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][0:3000][()])
                
                    ra_test_GW170817 = 2.0*np.pi*f_test_GW170817['ra'][()] # 35000
                    dec_test_GW170817 = np.arcsin(1.0-2.0*f_test_GW170817['dec'][()])
                
                    mass_1 = f_test['mass1'][0:3000][()]
                    mass_2 = f_test['mass2'][0:3000][()]
                    spin_1 = f_test['spin1z'][0:3000][()]
                    spin_2 = f_test['spin2z'][0:3000][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][0:3000][()]
    
                    mass_1_GW170817 = f_test_GW170817['mass1'][()]
                    mass_2_GW170817 = f_test_GW170817['mass2'][()]
                    spin_1_GW170817 = f_test_GW170817['spin1z'][()]
                    spin_2_GW170817 = f_test_GW170817['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_GW170817 = f_test_GW170817['Injection_SNR'][()]
                    
                    ra_test = np.concatenate([ra_test, ra_test_GW170817], axis=0)
                    dec_test = np.concatenate([dec_test, dec_test_GW170817], axis=0)
                    mass_1 = np.concatenate([mass_1, mass_1_GW170817], axis=0)
                    mass_2 = np.concatenate([mass_2, mass_2_GW170817], axis=0)
                    spin_1 = np.concatenate([spin_1, spin_1_GW170817], axis=0)
                    spin_2 = np.concatenate([spin_2, spin_2_GW170817], axis=0)
                    inj_snr = np.concatenate([inj_snr, inj_snr_GW170817], axis=0)        
            
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
            
                    f_test.close()
                    f_test_GW170817.close()
                    
                elif(data_config.train.test_negative_latency_seconds == '45'):
              
                    f_test = h5py.File(data_config.parameters.BNS.path_test_design_45_secs, 'r')
                    f_test_GW170817 = h5py.File(data_config.parameters.BNS.path_test_GW170817_45_secs, 'r')
           
                    ra_test = 2.0*np.pi*f_test['ra'][0:3000][()] # 8000
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][0:3000][()])
                
                    ra_test_GW170817 = 2.0*np.pi*f_test_GW170817['ra'][()] # 35000
                    dec_test_GW170817 = np.arcsin(1.0-2.0*f_test_GW170817['dec'][()])
                
                    mass_1 = f_test['mass1'][0:3000][()]
                    mass_2 = f_test['mass2'][0:3000][()]
                    spin_1 = f_test['spin1z'][0:3000][()]
                    spin_2 = f_test['spin2z'][0:3000][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][0:3000][()]
    
                    mass_1_GW170817 = f_test_GW170817['mass1'][()]
                    mass_2_GW170817 = f_test_GW170817['mass2'][()]
                    spin_1_GW170817 = f_test_GW170817['spin1z'][()]
                    spin_2_GW170817 = f_test_GW170817['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_GW170817 = f_test_GW170817['Injection_SNR'][()]
                    
                    ra_test = np.concatenate([ra_test, ra_test_GW170817], axis=0)
                    dec_test = np.concatenate([dec_test, dec_test_GW170817], axis=0)
                    mass_1 = np.concatenate([mass_1, mass_1_GW170817], axis=0)
                    mass_2 = np.concatenate([mass_2, mass_2_GW170817], axis=0)
                    spin_1 = np.concatenate([spin_1, spin_1_GW170817], axis=0)
                    spin_2 = np.concatenate([spin_2, spin_2_GW170817], axis=0)
                    inj_snr = np.concatenate([inj_snr, inj_snr_GW170817], axis=0)       
            
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
            
                    f_test.close() 
                    
                elif(data_config.train.test_negative_latency_seconds == '58'):
              
                    f_test = h5py.File(data_config.parameters.BNS.path_test_design_58_secs, 'r')
                    f_test_GW170817 = h5py.File(data_config.parameters.BNS.path_test_GW170817_58_secs, 'r')
           
                    ra_test = 2.0*np.pi*f_test['ra'][0:3000][()] # 7000
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][0:3000][()])
                
                    ra_test_GW170817 = 2.0*np.pi*f_test_GW170817['ra'][()] # 35000
                    dec_test_GW170817 = np.arcsin(1.0-2.0*f_test_GW170817['dec'][()])
                
                    mass_1 = f_test['mass1'][0:3000][()]
                    mass_2 = f_test['mass2'][0:3000][()]
                    spin_1 = f_test['spin1z'][0:3000][()]
                    spin_2 = f_test['spin2z'][0:3000][()]
#                    inc = f_test['inclination'][()]
                    inj_snr = f_test['Injection_SNR'][0:3000][()]
    
                    mass_1_GW170817 = f_test_GW170817['mass1'][()]
                    mass_2_GW170817 = f_test_GW170817['mass2'][()]
                    spin_1_GW170817 = f_test_GW170817['spin1z'][()]
                    spin_2_GW170817 = f_test_GW170817['spin2z'][()]
#                    inc = f_test['inclination'][()]
                    inj_snr_GW170817 = f_test_GW170817['Injection_SNR'][()]
                    
                    ra_test = np.concatenate([ra_test, ra_test_GW170817], axis=0)
                    dec_test = np.concatenate([dec_test, dec_test_GW170817], axis=0)
                    mass_1 = np.concatenate([mass_1, mass_1_GW170817], axis=0)
                    mass_2 = np.concatenate([mass_2, mass_2_GW170817], axis=0)
                    spin_1 = np.concatenate([spin_1, spin_1_GW170817], axis=0)
                    spin_2 = np.concatenate([spin_2, spin_2_GW170817], axis=0)
                    inj_snr = np.concatenate([inj_snr, inj_snr_GW170817], axis=0)       
            
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
            
                    f_test.close() 
                    
                                   
               
            
            elif((data_config.train.dataset == 'BNS') and (data_config.train.PSD == 'O4')):
                
                if(data_config.train.test_negative_latency_seconds == '0'):
                    
                    f_test = h5py.File(data_config.parameters.BNS.path_test_O4_PSD_0_sec, 'r')
                    ra_test = 2.0*np.pi*f_test['ra'][()]
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][()])
                    
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
                    
                    f_test.close()
            
                elif(data_config.train.test_negative_latency_seconds == '5'):
                    
                    f_test = h5py.File(data_config.parameters.BNS.path_test_O4_PSD_5_sec, 'r')
                    ra_test = 2.0*np.pi*f_test['ra'][()]
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][()])
                    
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
                    
                    f_test.close()
                    
                elif(data_config.train.test_negative_latency_seconds == '10'):
                    
                    f_test = h5py.File(data_config.parameters.BNS.path_test_O4_PSD_10_sec, 'r')
                    ra_test = 2.0*np.pi*f_test['ra'][()]
                    dec_test = np.arcsin(1.0-2.0*f_test['dec'][()])
                    
                    ra_test = ra_test - np.pi
                    ra_test_x = np.cos(ra_test)
                    ra_test_y = np.sin(ra_test)
                    
                    f_test.close()                                
            
        
        ra_test = ra_test[:,None]
        ra_test_x = ra_test_x[:, None]
        ra_test_y = ra_test_y[:, None]
        
        dec_test = dec_test[:,None]
        
        mass_1 = mass_1[:,None]
        mass_2= mass_2[:,None]
        spin_1 = spin_1[:,None]
        spin_2 = spin_2[:,None]

        y_test = np.concatenate((ra_test_x, ra_test_y, dec_test), axis=1).astype('float32')
        
        intrinsic_params = np.concatenate((mass_1, mass_2, spin_1, spin_2), axis=1)

        return y_test, ra_test, dec_test, gps_time, intrinsic_params
    
    @staticmethod
    def load_test_2_det_parameters(data_config):
        """Loads train parameters from path"""
        if((data_config.train.dataset == 'BNS') and (data_config.train.snr_range_test == 'low')):
            
            f_test_1 = h5py.File(data_config.parameters.BNS.path_test_2_det_low_SNR, 'r')
            f_test_2 = h5py.File(data_config.parameters.BNS.path_test, 'r')
            f_test_3 = h5py.File(data_config.parameters.BNS.path_test_2_det_high_SNR, 'r')
            
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
            
        if((data_config.train.dataset == 'BNS') and (data_config.train.test_negative_latency == True)):
            
#            f_test_1 = h5py.File(data_config.parameters.BNS.path_test_2_det_0_secs, 'r')
            f_test_2 = h5py.File(data_config.parameters.BNS.path_test_2_det_0_secs, 'r')
            f_test_3 = h5py.File(data_config.parameters.BNS.path_test_2_det_0_secs, 'r')
            f_test_4 = h5py.File(data_config.parameters.BNS.path_test_2_det_0_secs, 'r')
            
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
            
        elif((data_config.train.dataset == 'NSBH') and (data_config.train.snr_range_test == 'low')):
            
            f_test_1 = h5py.File(data_config.parameters.NSBH.path_test, 'r')
            f_test_2 = h5py.File(data_config.parameters.NSBH.path_test_low_snr, 'r')
            
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
            
        elif((data_config.train.dataset == 'BBH') and (data_config.train.test_real == False) and (data_config.train.snr_range_test == 'low')):
            
            f_test_1 = h5py.File(data_config.parameters.BBH.path_train, 'r')
            f_test_2 = h5py.File(data_config.parameters.BBH.path_test_low_SNR, 'r')
            
            ra_1 = 2.0*np.pi*f_test_1['ra'][0:2000][()]
            dec_1 = np.arcsin(1.0-2.0*f_test_1['dec'][0:2000][()])
            
            ra_2 = 2.0*np.pi*f_test_2['ra'][()]
            dec_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
                        
            ra_test = np.concatenate([ra_1, ra_2], axis=0)
            ra_test = ra_test - np.pi
            ra_test_x = np.cos(ra_test)
            ra_test_y = np.sin(ra_test)
            
            dec_test = np.concatenate([dec_1, dec_2], axis=0)

            f_test_1.close()
            f_test_2.close()

            
        elif((data_config.train.dataset == 'BBH') and (data_config.train.test_real == True) and (data_config.train.psd == 'O2')):
            
            f_test_1 = h5py.File(data_config.parameters.BBH.path_test_GW150914, 'r')
            f_test_2 = h5py.File(data_config.parameters.BBH.path_test_GW170104, 'r')
                    
            ra_test_1 = 2.0*np.pi*f_test_1['ra'][()] # 7000
            dec_test_1 = np.arcsin(1.0-2.0*f_test_1['dec'][()])
                
            ra_test_2 = 2.0*np.pi*f_test_2['ra'][()] # 35000
            dec_test_2 = np.arcsin(1.0-2.0*f_test_2['dec'][()])
                
            mass_1_1 = f_test_1['mass1'][()]
            mass_2_1 = f_test_1['mass2'][()]
            spin_1_1 = f_test_1['spin1z'][()]
            spin_2_1 = f_test_1['spin2z'][()]
#                    inc = f_test['inclination'][()]
            inj_snr_1 = f_test_1['Injection_SNR'][()]
            gps_time_1 = f_test_1['gps_time'][()]
    
            mass_1_2 = f_test_2['mass1'][()]
            mass_2_2 = f_test_2['mass2'][()]
            spin_1_2 = f_test_2['spin1z'][()]
            spin_2_2 = f_test_2['spin2z'][()]
#                    inc = f_test['inclination'][()]
            inj_snr_2 = f_test_2['Injection_SNR'][()]
            gps_time_2 = f_test_2['gps_time'][()]
                    
            ra_test = np.concatenate([ra_test_1, ra_test_2], axis=0)
            dec_test = np.concatenate([dec_test_1, dec_test_2], axis=0)
            mass_1 = np.concatenate([mass_1_1, mass_1_2], axis=0)
            mass_2 = np.concatenate([mass_2_1, mass_2_2], axis=0)
            spin_1 = np.concatenate([spin_1_1, spin_1_2], axis=0)
            spin_2 = np.concatenate([spin_2_1, spin_2_2], axis=0)
            inj_snr = np.concatenate([inj_snr_1, inj_snr_2], axis=0)
            gps_time = np.concatenate([gps_time_1, gps_time_2], axis=0)
            
            ra_test = ra_test - np.pi
            ra_test_x = np.cos(ra_test)
            ra_test_y = np.sin(ra_test)
        
        ra_test = ra_test[:,None]
        ra_test_x = ra_test_x[:, None]
        ra_test_y = ra_test_y[:, None]
        
        dec_test = dec_test[:,None]
        
        mass_1 = mass_1[:,None]
        mass_2= mass_2[:,None]
        spin_1 = spin_1[:,None]
        spin_2 = spin_2[:,None]

        y_test = np.concatenate((ra_test_x, ra_test_y, dec_test), axis=1).astype('float32')
        
        intrinsic_params = np.concatenate((mass_1, mass_2, spin_1, spin_2), axis=1)
        
        return y_test, ra_test, dec_test, gps_time, intrinsic_params
    
    
    @staticmethod
    def load_valid_samples(data_config, X_real, X_imag, y, intrinsic, data):
        """Loads 3 det pre-merger samples and parameters from path"""
        if((data_config.train.dataset == 'BNS') or (data_config.train.dataset == 'NSBH') or (data_config.train.dataset == 'BBH')):
            
            max_snr_h1 = np.max(abs(X_real[:,:,0]), axis=1)
            max_snr_l1 = np.max(abs(X_real[:,:,1]), axis=1)
            
            if(data_config.train.num_detectors == 3):
                max_snr_v1 = np.max(abs(X_real[:,:,2]), axis=1)
            
            dec = y[:,2]
            
            if(data_config.train.num_detectors == 3):
                net_snr = np.sqrt(max_snr_h1**2 + max_snr_l1**2 + max_snr_v1**2)
            else:
                net_snr = np.sqrt(max_snr_h1**2 + max_snr_l1**2)
            
            if(data == 'train'):
                valid_h1 = max_snr_h1 >= 3   # 4
                valid_l1 = max_snr_l1 >= 3  # 4
                
                if(data_config.train.num_detectors == 3):
                    valid_v1 = max_snr_v1 >= 1   # 4
                    
                valid_net_snr = ((net_snr >= data_config.train.min_snr) & (net_snr <= 100))  # 40

                if(data_config.train.num_detectors == 3):
                    h1l1 = np.bitwise_and(valid_h1, valid_l1)
                    l1v1 = np.bitwise_and(valid_l1, valid_v1)
                    h1v1 = np.bitwise_and(valid_h1, valid_v1)
                    
                    h1l1v1 = np.bitwise_and(h1l1, valid_v1)

                    h1l1_or_l1v1_or_h1v1 = np.bitwise_or((np.bitwise_or(h1l1,h1v1)),l1v1)
                    h1l1_or_l1v1_or_h1v1_and_net_snr = np.bitwise_and(h1l1_or_l1v1_or_h1v1, valid_net_snr)
                
                    valid_sample = h1l1_or_l1v1_or_h1v1_and_net_snr
                    
                elif(data_config.train.num_detectors == 2):
                    h1l1 = np.bitwise_and(valid_h1, valid_l1)
                    h1l1_and_net_snr = np.bitwise_and(h1l1, valid_net_snr)
                    
                    valid_sample = h1l1_and_net_snr
                
                
            elif((data == 'test') and (data_config.train.train_negative_latency_seconds!='0')):
                
                valid_h1 = max_snr_h1 >= 3   # 4
                valid_l1 = max_snr_l1 >= 3   # 4
                
                if(data_config.train.num_detectors == 3):
                    valid_v1 = max_snr_v1 >= 3   # 4
                
                valid_net_snr = ((net_snr >= data_config.train.min_snr) & (net_snr <= 50))  # 40


                if(data_config.train.num_detectors==3):
        
                    h1l1 = np.bitwise_and(valid_h1, valid_l1)
                    l1v1 = np.bitwise_and(valid_l1, valid_v1)
                    h1v1 = np.bitwise_and(valid_h1, valid_v1)

                    h1l1_or_l1v1_or_h1v1 = np.bitwise_or((np.bitwise_or(h1l1,h1v1)),l1v1)
                    h1l1_or_l1v1_or_h1v1_and_net_snr = np.bitwise_and(h1l1_or_l1v1_or_h1v1, valid_net_snr)
                
                    valid_sample = h1l1_or_l1v1_or_h1v1_and_net_snr
               
                elif(data_config.train.num_detectors==2):
        
                    h1l1 = np.bitwise_and(valid_h1, valid_l1)
                    
                    h1l1_and_net_snr = np.bitwise_and(h1l1, valid_net_snr)
                
                    valid_sample = h1l1_and_net_snr    

            elif((data == 'test') and (data_config.train.test_negative_latency_seconds == '0')): 
                
########################################### For everything except 0 sec pre-merger #############################################

                valid_h1 = max_snr_h1 > 0.0   # 3
                valid_l1 = max_snr_l1 > 0.0   # 3
        
                if(data_config.train.num_detectors == 3):
                    valid_v1 = max_snr_v1 > 0.0   # 1
                
#                valid_net_snr = ((net_snr >= 8) & (net_snr <= 50))  # 40
                valid_net_snr = net_snr >= 0.0  # 40

            
                if(data_config.train.num_detectors == 3):
            
                    h1l1 = np.bitwise_and(valid_h1, valid_l1)
                    l1v1 = np.bitwise_and(valid_l1, valid_v1)
                    h1v1 = np.bitwise_and(valid_h1, valid_v1)
                    
                    h1l1v1 = np.bitwise_and(h1l1, valid_v1)
                
                    h1l1_or_l1v1_or_h1v1 = np.bitwise_or((np.bitwise_or(h1l1,h1v1)),l1v1)
                    h1l1_or_l1v1_or_h1v1_and_net_snr = np.bitwise_and(h1l1_or_l1v1_or_h1v1, valid_net_snr)
                
                    valid_sample = h1l1_or_l1v1_or_h1v1_and_net_snr
                    
                elif(data_config.train.num_detectors == 2):
            
                    h1l1 = np.bitwise_and(valid_h1, valid_l1)
                
                    h1l1_and_net_snr = np.bitwise_and(h1l1, valid_net_snr)
                
                    valid_sample = h1l1_and_net_snr    

######################################## Only for 0 sec pre-merger #############################################################

                
#                valid_h1 = max_snr_h1 > 3   # 4
#                valid_l1 = max_snr_l1 > 3   # 4
#                valid_v1 = max_snr_v1 > 3   # 4
#                valid_net_snr_1 = ((net_snr > 8) & (net_snr <= 12))  # 40

#                h1l1 = np.bitwise_and(valid_h1, valid_l1)
#                l1v1 = np.bitwise_and(valid_l1, valid_v1)
#                h1v1 = np.bitwise_and(valid_h1, valid_v1)

#                h1l1_or_l1v1_or_h1v1 = np.bitwise_or((np.bitwise_or(h1l1,h1v1)),l1v1)
#                h1l1_or_l1v1_or_h1v1_and_net_snr_1 = np.bitwise_and(h1l1_or_l1v1_or_h1v1, valid_net_snr_1)
            
#                valid_sample_1 = np.where(h1l1_or_l1v1_or_h1v1_and_net_snr_1 == True)

#                valid_h1 = max_snr_h1 > 3   # 4
#                valid_l1 = max_snr_l1 > 3   # 4
#                valid_v1 = max_snr_v1 > 3   # 4
#                valid_net_snr_2 = ((net_snr > 12) & (net_snr <= 40))  # 40

#                h1l1 = np.bitwise_and(valid_h1, valid_l1)
#                l1v1 = np.bitwise_and(valid_l1, valid_v1)
#                h1v1 = np.bitwise_and(valid_h1, valid_v1)

#                h1l1_or_l1v1_or_h1v1 = np.bitwise_or((np.bitwise_or(h1l1,h1v1)),l1v1)
#                h1l1_or_l1v1_or_h1v1_and_net_snr_2 = np.bitwise_and(h1l1_or_l1v1_or_h1v1, valid_net_snr_2)
            
#                valid_sample_2 = np.where(h1l1_or_l1v1_or_h1v1_and_net_snr_2 == True)

#                valid_sample = np.append(valid_sample_1[0], valid_sample_2[0][0:2000])
##                valid_sample = np.append(valid_sample_1[0], valid_sample_2[0]) # only for testing SNR - 500. Comment out otherwise.
                
######################################### For splitting high and low SNR during testing ########################################   
#                valid_net_snr_3 = ((net_snr > 25) & (net_snr <= 40))  # 40
#                h1l1_or_l1v1_or_h1v1 = np.bitwise_or((np.bitwise_or(h1l1,h1v1)),l1v1)
#                h1l1_or_l1v1_or_h1v1_and_net_snr_3 = np.bitwise_and(h1l1_or_l1v1_or_h1v1, valid_net_snr_3)
#                valid_sample_3 = np.where(h1l1_or_l1v1_or_h1v1_and_net_snr_3 == True)
#                valid_high_SNR = np.intersect1d(valid_sample, valid_sample_3)
#                valid_low_SNR = np.setxor1d(valid_high_SNR, valid_sample)
                
#                if(self.min_snr == 25):
#                    valid_sample = valid_high_SNR
                    
#                elif(self.min_snr == 8):
#                    valid_sample = valid_low_SNR

################################################################################################################################

            X_real = X_real[valid_sample]
            X_imag = X_imag[valid_sample]
            y_samples = y[valid_sample]
            ra_x = y[:,0][valid_sample]
            ra_y = y[:,1][valid_sample]
            ra = np.arctan2(ra_y, ra_x)
            dec = y[:,2][valid_sample]
            intrinsic = intrinsic[valid_sample]
            net_snr = net_snr[valid_sample]
                
            return X_real, X_imag, y_samples, ra, ra_x, ra_y, dec, intrinsic, valid_sample, net_snr 
                
                
