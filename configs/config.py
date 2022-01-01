"""Model config in json format"""

CFG = {
    "data": {
        
        "NSBH": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_52k.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_30k.hdf",
                "path_train_3": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_42-47.hdf",
                "path_train_4": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_train_48-50.hdf",
            
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_NSBH_test.hdf"
                },        
        "BBH": {
                "path_train": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_training_200k.hdf",
                },
        "BNS": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_1-6.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_7-24.hdf",
            
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_test.hdf"
                },
    },
    "parameters": {
    
        "NSBH": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_52k.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_30k.hdf",
                "path_train_3": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_42-47.hdf",
                "path_train_4": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_injection_run_parameters_NSBH_train_48-50.hdf",
        
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_NSBH_test.hdf"
                },
        "BBH": {
                "path_train": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_200k_injection_parameters",
            
                },
        "BNS": {
                "path_train_1": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_1-6.hdf",
                "path_train_2": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_7-24.hdf",
            
                "path_test": "/group/pmc005/cchatterjee/SNR_time_series_sample_files/default_GW170817_parameters_BNS_test.hdf"
                },   
    },
    "train": {
            "network": "ResNet",
            "dataset": "BNS",
            "num_train": 108000,
            "num_test": 2000,
            "min_snr": 6,
            "n_samples": 410,
            "batch_size": 2000,
            "num_detectors": 3,
            "epochs": 75,
            "validation_split": 0.05,
            "optimizer": {
                "type": "adam"
            },
    },
    "model": { # best: num_bijectors: 6, MAF_hidden_units: [256, 256, 256], epochs: 50.
            "num_bijectors": 6,
            "MAF_hidden_units": [64,64,64],
        
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
            "learning_rate": 1e-4
    }
}
