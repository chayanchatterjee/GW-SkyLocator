"""Model config in json format"""

CFG = {
    "data": {
        
        "NSBH": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_52k.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_30k.hdf",
                "path_train_3": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_42-47.hdf",
                "path_train_4": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_48-50.hdf",
            
                "path_train_low_snr_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_snr-10to20_train_4k_samples.hdf",
                "path_train_low_snr_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_snr-10to20_NSBH_train_OzSTAR_train.hdf",
            
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_test.hdf",
                "path_test_low_snr": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_snr-10to20_NSBH_test.hdf",
                },
        
        "BBH": {
                "path_train": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_training_200k.hdf",
                
                "path_train_low_SNR": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_snr-10to20_train.hdf",
            
                "path_test_low_SNR": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_snr-10to20_test.hdf"
                },
        
        "BNS": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_1-6.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_7-24.hdf",
                
                "path_train_low_snr_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_snr-10to20_train_2k_samples.hdf",
                "path_train_low_snr_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_snr-10to20_train_4k_samples.hdf",
                "path_train_low_snr_3": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_snr-10to20_train_OzSTAR.hdf",
            
                "path_train_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_2_det_snr-10to20_low_mass.hdf",
                
            
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_test.hdf",
                "path_test_low_SNR": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_snr_10-20_test.hdf",
                "path_test_GW170817": "/group/pmc005/cchatterjee/Real_events/default_snr_series_GW170817_test_Gaussian_noise_1.hdf",
            
                "path_test_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_2_det_snr_10-20_BNS_test.hdf",
                },
    },
    "parameters": {
    
        "NSBH": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_52k.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_30k.hdf",
                "path_train_3": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_42-47.hdf",
                "path_train_4": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_48-50.hdf",
            
                "path_train_low_snr_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_NSBH_train_snr-10to20_4k_samples.hdf",
                "path_train_low_snr_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_snr-10to20_NSBH_train_OzSTAR_parameters.hdf",
        
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_NSBH_test.hdf",
                "path_test_low_snr": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_snr-10to20_NSBH_test_parameters.hdf"
            
                },
        
        "BBH": {
                "path_train": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_200k_injection_parameters",
                "path_train_low_SNR": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BBH_train_snr-10to20.hdf",
                "path_test_low_SNR": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_BBH_snr-10to20_test_parameters.hdf"
            
                },
        
        "BNS": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_1-6.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_7-24.hdf",
            
                "path_train_low_snr_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_snr-10to20_2k_samples.hdf",
                "path_train_low_snr_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_snr-10to20_4k_samples.hdf",
                "path_train_low_snr_3": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_snr-10to20_train_OzSTAR_parameters.hdf",
            
                "path_train_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_BNS_2_det_snr-10to20_low_mass_parameters.hdf",
                
            
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_test.hdf",
                "path_test_low_SNR": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_test_snr_10-20_BNS_test.hdf",
                "path_test_GW170817": "/group/pmc005/cchatterjee/Real_events/default_GW170817_parameters_test_Gaussian_noise_1.hdf",
                
                "path_test_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_2_det_snr_10-20_BNS_test_parameters.hdf",
                
                },   
    },
    "train": {
            "network": "ResNet-34_2_det",
            "dataset": "BNS",
            "test_real": False,
            "snr_range_train": 'low',
            "snr_range_test": 'low',
            "num_train": 100000,
            "num_test": 2000,
            "min_snr": 4,
            "n_samples": 410,
            "batch_size": 2000,
            "output_filename": 'Adaptive_NSIDE/Injection_run_BNS_2_det_snr-10to20_low_mass.hdf',
            "num_detectors": 2,
                "epochs": 100,
            "validation_split": 0.05,
            "optimizer": {
                "type": "adam"
            },
    },
    "model": { # best: num_bijectors: 6, MAF_hidden_units: [256, 256, 256], epochs: 50.
            "num_bijectors": 6,
            "MAF_hidden_units": [2048, 2048, 2048, 2048, 2048],
        
            "WaveNet": {
                        "filters": 16,
                        "kernel_size": 3,
                        "activation": "relu",
                        "dilation_rate": 1
            },
            "ResNet": {
                        "kernels_resnet_block": 32,
                        "stride_resnet_block" : 1,
                        "kernel_size_resnet_block": 3,
                        "kernels": 32,
                        "kernel_size": 5,
                        "strides": 3
            
            },
            "ResNet_34": {
                        "filters": 32,
                        "kernel_size" : 7,
                        "strides": 2,
                        "pool_size": 3,
                        "prev_filters": 32
            
            },
            "ResNet_34_2_det": {
                        "filters": 32,
                        "kernel_size" : 7,
                        "strides": 2,
                        "pool_size": 2,
                        "prev_filters": 32
            
            },
            "LSTM_model": {
                        "n_units": 128,
                        "rate": 0.20
            
            },
            "CNN_model": {
                        "filters": 32,
                        "kernel_size": 5,
                        "max_pool_size": 2,
                        "dropout_rate": 0.2,
                        "n_units": 128
            
            },
            "learning_rate": 1e-4
    }
}
