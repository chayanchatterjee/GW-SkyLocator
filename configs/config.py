"""Model config in json format"""

CFG = {
    "data": {
        
        "NSBH": {
                "path_train_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_NSBH_train_52k.hdf",
                "path_train_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_NSBH_train_30k.hdf",
                "path_train_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_NSBH_train_42-47.hdf",
                "path_train_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_NSBH_train_48-50.hdf",
            
                "path_train_low_snr_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_NSBH_snr-10to20_train_4k_samples.hdf",
                "path_train_low_snr_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_snr-10to20_NSBH_train_OzSTAR_train.hdf",
                "path_train_low_snr_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_NSBH_2_det_snr-20to30_72k.hdf",
            
                "path_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_NSBH_test.hdf",
                "path_test_low_snr": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_snr-10to20_NSBH_test.hdf",
                },
        
        "BBH": {
                "path_train": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_training_200k.hdf",
                
                "path_train_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_snr-10to20_train.hdf",
            
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_snr-10to20_test.hdf"
                },
        
        "BNS": {
                "path_train_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_1-6.hdf",
                "path_train_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_7-24.hdf",
                
                "path_train_low_snr_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_snr-10to20_train_2k_samples.hdf",
                "path_train_low_snr_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_snr-10to20_train_4k_samples.hdf",
                "path_train_low_snr_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_snr-10to20_train_OzSTAR.hdf",
            
                "path_train_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_2_det_snr-10to20_low_mass.hdf",
                "path_train_2_det_low_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr_10-20_new.hdf",
            
                "path_train_2_det_high_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr-30to40_train_1.hdf",
                "path_train_2_det_high_SNR_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr-30to40_train_2.hdf",
                
            
                "path_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_test.hdf",
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_snr_10-20_test.hdf",
                "path_test_GW170817": "/group/pmc005/cchatterjee/Real_events/default_snr_series_GW170817_test_Gaussian_noise_1.hdf",
            
                "path_test_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_2_det_snr_10-20_BNS_test.hdf",
                
                "path_test_2_det_high_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr-30to40_test.hdf",
            
                "path_test_3_det_0_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_0_secs_negative.hdf",
                "path_test_3_det_5_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_5_secs_negative.hdf",
                "path_test_3_det_10_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_10_secs_negative.hdf",
                "path_test_3_det_15_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_15_secs_negative.hdf",
                
            
                "path_test_2_det_0_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_0_secs_negative.hdf",
                "path_test_2_det_5_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_5_secs_negative.hdf",
                "path_test_2_det_10_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_10_secs_negative.hdf",
                "path_test_2_det_15_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_15_secs_negative.hdf"
            
                },
    },
    "parameters": {
    
        "NSBH": {
                "path_train_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_injection_run_parameters_NSBH_train_52k.hdf",
                "path_train_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_injection_run_parameters_NSBH_train_30k.hdf",
                "path_train_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_injection_run_parameters_NSBH_train_42-47.hdf",
                "path_train_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_injection_run_parameters_NSBH_train_48-50.hdf",
            
                "path_train_low_snr_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_parameters_NSBH_train_snr-10to20_4k_samples.hdf",
                "path_train_low_snr_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_snr-10to20_NSBH_train_OzSTAR_parameters.hdf",
                "path_train_low_snr_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_NSBH_2_det_parameters_snr-20to30_72k.hdf",
            
        
                "path_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_parameters_NSBH_test.hdf",
                "path_test_low_snr": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_snr-10to20_NSBH_test_parameters.hdf"
            
                },
        
        "BBH": {
                "path_train": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_200k_injection_parameters",
                "path_train_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BBH_train_snr-10to20.hdf",
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_BBH_snr-10to20_test_parameters.hdf"
            
                },
        
        "BNS": {
                "path_train_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_1-6.hdf",
                "path_train_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_7-24.hdf",
            
                "path_train_2_det_high_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_snr-30to40_parameters_1.hdf",
                "path_train_2_det_high_SNR_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_snr-30to40_parameters_2.hdf",
                
            
                "path_train_low_snr_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_parameters_BNS_train_snr-10to20_2k_samples.hdf",
                "path_train_low_snr_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_parameters_BNS_train_snr-10to20_4k_samples.hdf",
                "path_train_low_snr_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_snr-10to20_train_OzSTAR_parameters.hdf",
            
                "path_train_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_BNS_2_det_snr-10to20_low_mass_parameters.hdf",
                "path_train_2_det_low_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_parameters_BNS_train_snr-10to20_new.hdf",
                            
            
                "path_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BNS_test.hdf",
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_test_snr_10-20_BNS_test.hdf",
                "path_test_GW170817": "/group/pmc005/cchatterjee/Real_events/default_GW170817_parameters_test_Gaussian_noise_1.hdf",
            
                "path_test_2_det_high_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_snr-30to40_parameters_test.hdf",
                
                "path_test_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_2_det_snr_10-20_BNS_test_parameters.hdf",
                            
                "path_test_3_det_0_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_negative_latency_0_sec_parameters.hdf",
            
                "path_test_2_det_0_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_2_det_0_sec_parameters.hdf"
                
                },   
    },
    "train": {
            "network": "ResNet-34_2_det",
            "dataset": "NSBH",
            "test_real": False,
            "test_negative_latency": False,
            "snr_range_train": 'low',
            "snr_range_test": 'low',
            "num_train": 272000, # BNS 2 det: 410000, NSBH 2 det: 272000, BBH 2 det: 280000
            "num_test": 4000,
            "min_snr": 4,
            "n_samples": 410,
            "batch_size": 2000,
            "output_filename": 'Adaptive_NSIDE/Injection_run_NSBH_2_det_snr-10to20_new.hdf',
            "num_detectors": 2,
                "epochs": 55, # For BNS: 75, for NSBH: 100
            "validation_split": 0.05,
            "optimizer": {
                "type": "adam"
            },
    },
    "model": { # best: num_bijectors: 6, MAF_hidden_units: [256, 256, 256], epochs: 50.
               # best 2 detectors: num_bijectors: 6, MAF_hidden_units: [2048, 2048, 2048, 2048, 2048], epochs=75
            "num_bijectors": 6,
            "MAF_hidden_units": [1024, 1024, 1024, 1024, 1024],
        
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
                        "filters_real": 32,
                        "filters_imag": 32, # 8
                        "kernel_size" : 7,
                        "strides": 2,
                        "pool_size": 3,
                        "prev_filters_real": 32,
                        "prev_filters_imag": 32 # 8
            
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
