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
            
                "path_train_design_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_NSBH_3_det_design_0_sec_new_1.hdf",
                "path_train_design_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_NSBH_3_det_design_0_sec_new_2.hdf",
                "path_train_design_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_NSBH_3_det_design_0_sec_new_3.hdf",
                "path_train_design_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_NSBH_3_det_design_0_sec_new_4.hdf",
                "path_train_design_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_NSBH_3_det_design_0_sec_new_5.hdf",
            
            
                "path_train_real_noise_1":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_1/O2_noise_snr_series_GW170817_NSBH_3_det_0_secs_1.hdf",
                "path_train_real_noise_2":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_2/O2_noise_snr_series_GW170817_NSBH_3_det_0_secs_2.hdf",
                "path_train_real_noise_3":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_3/O2_noise_snr_series_GW170817_NSBH_3_det_0_secs_3.hdf",
                "path_train_real_noise_4":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_4/O2_noise_snr_series_GW170817_NSBH_3_det_0_secs_4.hdf",
                "path_train_real_noise_5":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_5/O2_noise_snr_series_GW170817_NSBH_3_det_0_secs_5.hdf",
            
                "path_test_Bayestar_post_merger_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_NSBH_10.hdf",
                "path_test_Bayestar_post_merger_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_NSBH_12.hdf",
                "path_test_Bayestar_post_merger_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_2.hdf",
                "path_test_Bayestar_post_merger_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_996.hdf",
                "path_test_Bayestar_post_merger_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_BBH_100.hdf",
                "path_test_Bayestar_post_merger_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_BBH_20.hdf",
                "path_test_Bayestar_post_merger_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_BBH_19.hdf",
            
                "path_test_GW190814": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_snr_time_series_GW190814.hdf", 
            
                "path_test_design_Bayestar_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_NSBH_3_det_design_0_sec_test_Bayestar_1.hdf",
            
                "path_test_Bayestar_post_merger": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_NSBH_10.hdf",
            
                "path_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_NSBH_test.hdf",
                "path_test_low_snr": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_snr-10to20_NSBH_test.hdf",
                "path_test_GW190917": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW190917_test_new_1.hdf",
                },
        
        "BBH": {
                "path_train": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_training_200k.hdf",
                
                "path_train_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_snr-10to20_train.hdf",
            
                "path_train_design_1":  "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BBH_3_det_design_0_sec_new_1.hdf",
                "path_train_design_2":  "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BBH_3_det_design_0_sec_new_2.hdf",
                "path_train_design_3":  "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BBH_3_det_design_0_sec_new_3.hdf",
                "path_train_design_4":  "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BBH_3_det_design_0_sec_new_4.hdf",
                "path_train_design_5":  "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BBH_3_det_design_0_sec_new_5.hdf",
            
                "path_train_real_noise_1":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_1/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_1.hdf",
                "path_train_real_noise_2":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_2/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_2.hdf",
                "path_train_real_noise_3":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_2/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_3.hdf",
                "path_train_real_noise_4":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_3/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_4.hdf",
                "path_train_real_noise_5":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_4/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_5.hdf",
            
                "path_train_real_noise_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_1/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_unequal_masses_1.hdf",
                "path_train_real_noise_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_2/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_unequal_masses_2.hdf",
                "path_train_real_noise_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_3/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_unequal_masses_3.hdf",
                "path_train_real_noise_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_4/O2_noise_snr_series_GW170817_BBH_3_det_0_secs_unequal_masses_4.hdf",
            
                "path_train_O3_noise_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_1/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_1.hdf",
                "path_train_O3_noise_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_2/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_2.hdf",
                "path_train_O3_noise_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_3/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_3.hdf",
                "path_train_O3_noise_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_4/O3_noise_snr_series_GW170817_BBH_3_det_0_secs_4.hdf",
                
                            
                "path_test_GW200224_222234": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_snr_time_series_GW200224_222234.hdf", 
                "path_test_GW190412": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_snr_time_series_GW190412.hdf",
                "path_test_GW150914": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_snr_time_series_GW150914.hdf",
                "path_test_GW170104": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_snr_time_series_GW170104.hdf",
                "path_test_GW190521": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_snr_time_series_GW190521.hdf",
            
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BBH_snr-10to20_test.hdf",
            
                "path_test_design_Bayestar_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BBH_3_det_design_0_sec_test_Bayestar_1.hdf",
            
                "path_test_Bayestar_post_merger": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_BBH_100.hdf",
            
                "path_test_GW170729":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170729_test_.hdf",
                "path_test_GW170809":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170809_test_.hdf",
                "path_test_GW170814":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170814_test_.hdf",
                "path_test_GW170818":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170818_test_.hdf",
            
                },
        
        "BNS": {
                "path_train_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_1-6.hdf",
                "path_train_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_train_7-24.hdf",
         # Real data       
#                "path_train_real_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/default_snr_series_GW170817_realnoise_1.hdf",
                "path_train_real_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/default_snr_series_GW170817_realnoise_final_Kaya.hdf",
                "path_train_real_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/default_snr_series_GW170817_realnoise_final_Pople.hdf",
         
         # Low SNR
                "path_train_low_snr_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_snr-10to20_train_2k_samples.hdf",
                "path_train_low_snr_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_snr-10to20_train_4k_samples.hdf",
                "path_train_low_snr_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_snr-10to20_train_OzSTAR.hdf",
            
        # Pre-merger
                "path_train_5_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_5_sec_train.hdf",
                "path_train_10_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_10_sec_train.hdf",
            
        # O4 PSD
                # 0 secs
                "path_train_O4_PSD_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_1.hdf",
                "path_train_O4_PSD_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_2.hdf",
                "path_train_O4_PSD_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_3.hdf",
            
                # 5 secs
                "path_train_O4_PSD_5_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_5_sec_1.hdf",
                "path_train_O4_PSD_5_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_5_sec_2.hdf",
                "path_train_O4_PSD_5_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_5_sec_3.hdf",
                "path_train_O4_PSD_5_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_5_sec_4.hdf",
                "path_train_O4_PSD_5_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_O4_PSD_5_sec_5.hdf",
            
       # Design Sensitivity
                # 0 secs
                "path_train_design_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_1.hdf",
                "path_train_design_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_2.hdf",
                "path_train_design_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_3.hdf",
                "path_train_design_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_4.hdf",
                "path_train_design_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_5.hdf",
                "path_train_design_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_6.hdf",
                "path_train_design_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_7.hdf",
                "path_train_design_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_8.hdf",
                "path_train_design_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_9.hdf",
                "path_train_design_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_10.hdf",
                
                "path_train_design_high_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_test_high_SNR_1.hdf",
                "path_train_design_high_SNR_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_test_high_SNR_2.hdf",
            
                "path_train_O2_noise_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_1/O2_noise_snr_series_GW170817_BNS_3_det_0_secs_1.hdf",
                "path_train_O2_noise_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_1/O2_noise_snr_series_GW170817_BNS_3_det_0_secs_2.hdf",
                "path_train_O2_noise_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_2/O2_noise_snr_series_GW170817_BNS_3_det_0_secs_3.hdf",
                "path_train_O2_noise_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_2/O2_noise_snr_series_GW170817_BNS_3_det_0_secs_4.hdf",
                "path_train_O2_noise_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_3/O2_noise_snr_series_GW170817_BNS_3_det_0_secs_5.hdf",
                
            
                "path_test_design_high_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_test_high_SNR.hdf",
                "path_test_design_Bayestar_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_test_Bayestar_1.hdf",
            
                "path_test_Bayestar_post_merger": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_snr_time_series_2.hdf",
            
            # 10 secs
                "path_train_design_10_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec.hdf",
                "path_train_design_10_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_1.hdf",
                "path_train_design_10_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_2.hdf",
                "path_train_design_10_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_3.hdf",
                "path_train_design_10_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_4.hdf",
                "path_train_design_10_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_5.hdf",
                "path_train_design_10_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_6.hdf",
                "path_train_design_10_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_7.hdf",
                "path_train_design_10_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_8.hdf",
                "path_train_design_10_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_9.hdf",
                "path_train_design_10_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_10.hdf",
                "path_train_design_10_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_11.hdf",
                "path_train_design_10_sec_13": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_12.hdf",
                "path_train_design_10_sec_14": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_13.hdf",
                "path_train_design_10_sec_15": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_14.hdf",
                "path_train_design_10_sec_16": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_15.hdf",
                "path_train_design_10_sec_17": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_16.hdf",
                "path_train_design_10_sec_18": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_17.hdf",
                "path_train_design_10_sec_19": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_18.hdf",
                "path_train_design_10_sec_20": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_19.hdf",
            
            # 15 secs
                "path_train_design_15_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_1.hdf",
                "path_train_design_15_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_2.hdf",
                "path_train_design_15_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_3.hdf",
                "path_train_design_15_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_4.hdf",
                "path_train_design_15_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_5.hdf",
                "path_train_design_15_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_6.hdf",
                "path_train_design_15_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_7.hdf",
                "path_train_design_15_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_8.hdf",
                "path_train_design_15_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_9.hdf",
                "path_train_design_15_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_10.hdf",
                "path_train_design_15_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_11.hdf",
                "path_train_design_15_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_12.hdf",
            
            # 30 secs
                "path_train_design_30_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_1.hdf",
                "path_train_design_30_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_2.hdf",
                "path_train_design_30_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_3.hdf",
                "path_train_design_30_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_4.hdf",
                "path_train_design_30_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_5.hdf",
                "path_train_design_30_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_train_6.hdf",
                "path_train_design_30_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_train_7.hdf",
                "path_train_design_30_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_train_8.hdf",
                "path_train_design_30_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_train_9.hdf",
                "path_train_design_30_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_train_10.hdf",
                "path_train_design_30_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_train_11.hdf",
                "path_train_design_30_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_train_12.hdf",
            
             # 45 secs
                "path_train_design_45_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_1.hdf",
            
                "path_train_design_45_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_2.hdf",
            
                "path_train_design_45_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_3.hdf",
            
                "path_train_design_45_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_4.hdf",
            
                "path_train_design_45_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_5.hdf",
            
            
                "path_train_design_45_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_train_6.hdf",
                "path_train_design_45_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_train_7.hdf",
                "path_train_design_45_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_train_8.hdf",
                "path_train_design_45_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_9.hdf",
                "path_train_design_45_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_10.hdf",
                "path_train_design_45_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_11.hdf",
            
                "path_train_design_45_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_1/default_snr_series_GW170817_BNS_3_det_45_secs_new_1.hdf",
                "path_train_design_45_sec_13": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_2/default_snr_series_GW170817_BNS_3_det_45_secs_new_2.hdf",
                "path_train_design_45_sec_14": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_3/default_snr_series_GW170817_BNS_3_det_45_secs_new_3.hdf",
                "path_train_design_45_sec_15": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_4/default_snr_series_GW170817_BNS_3_det_45_secs_new_4.hdf",
                "path_train_design_45_sec_16": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_4/default_snr_series_GW170817_BNS_3_det_45_secs_new_5.hdf",
            
            
            # 58 secs
                "path_train_design_58_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_1.hdf",
                "path_train_design_58_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_2.hdf",
                "path_train_design_58_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_3.hdf",
                "path_train_design_58_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_4.hdf",
                "path_train_design_58_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_train_5.hdf",
                "path_train_design_58_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_train_6.hdf",
                "path_train_design_58_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_train_7.hdf",
                "path_train_design_58_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_train_8.hdf",
                "path_train_design_58_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_train_9.hdf",
               
                           
            # Low SNR
                "path_train_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_BNS_2_det_snr-10to20_low_mass.hdf",
                "path_train_2_det_low_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr_10-20_new.hdf",
            
            # High SNR
                "path_train_2_det_high_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr-30to40_train_1.hdf",
                "path_train_2_det_high_SNR_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr-30to40_train_2.hdf",
             
           # Pre-merger 
                "path_train_2_det_5_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_2_det_5_sec_1_train.hdf",
                "path_train_2_det_5_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_2_det_5_sec_2_train.hdf",
                                
            
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_BNS_snr_10-20_test.hdf",
            
#                "path_test_GW170817": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_test_Gaussian_noise.hdf",
                "path_test_GW170817": "/fred/oz016/Chayan/samplegen/GW170817_real_data_glitch_removed/real_event_snr_time_series_GW170817_glitch_removed.hdf",
            
                "path_test_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_snr_series_GW170817_2_det_snr_10-20_BNS_test.hdf",
                
                "path_test_2_det_high_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_snr_series_GW170817_2_det_snr-30to40_test.hdf",
            
                "path_test_3_det_0_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_0_secs_negative.hdf",
                "path_test_3_det_5_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_aLIGO_PSD_5_sec_test.hdf",
                "path_test_3_det_10_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_aLIGO_PSD_10_sec_test.hdf",
                "path_test_3_det_15_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_15_secs_negative.hdf",
            
            
                "path_test_2_det_0_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_0_secs_negative.hdf",
                "path_test_2_det_5_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_5_secs_negative.hdf",
                "path_test_2_det_10_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_10_secs_negative.hdf",
                "path_test_2_det_15_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_2_det_15_secs_negative.hdf",
            
                "path_test_O4_PSD_0_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_O4_PSD_0_sec_test.hdf",
                "path_test_O4_PSD_5_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_O4_PSD_5_sec_test.hdf",
                "path_test_O4_PSD_10_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_O4_PSD_10_sec_test.hdf",
            

                "path_test_design": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_0_sec_40k_test_new.hdf",
                "path_test_GW170817_0_secs": "/fred/oz016/Chayan/samplegen/output/default_snr_series_test_GW170817_0_secs.hdf",

                "path_test_design_10_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_10_sec_test_low_SNR_new.hdf",
                "path_test_GW170817_10_secs": "/fred/oz016/Chayan/samplegen/output/default_snr_series_test_GW170817_10_secs.hdf",
            
                "path_test_design_15_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_15_sec_test_low_SNR_new.hdf",
                "path_test_GW170817_15_secs": "/fred/oz016/Chayan/samplegen/output/default_snr_series_test_GW170817_15_secs.hdf",
                
#                "path_test_design_30_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_8k_test_new.hdf",
                "path_test_design_30_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_30_sec_test_low_SNR_new.hdf",
                "path_test_GW170817_30_secs": "/fred/oz016/Chayan/samplegen/output/default_snr_series_test_GW170817_30_secs.hdf",
            
                "path_test_design_45_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_45_sec_test_low_SNR_new.hdf",
#                "path_test_design_45_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_1/default_snr_series_GW170817_BNS_3_det_45_secs_new_1.hdf",
            
                "path_test_GW170817_45_secs": "/fred/oz016/Chayan/samplegen/output/default_snr_series_test_GW170817_45_secs.hdf",
            
                "path_test_design_58_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_snr_series_GW170817_BNS_3_det_design_58_sec_test_low_SNR_new.hdf",
                "path_test_GW170817_58_secs": "/fred/oz016/Chayan/samplegen/output/default_snr_series_test_GW170817_58_secs.hdf",
                
                "path_test_design_example_0_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/example/default_snr_series_premerger_test_0_sec.hdf",
            
                "path_test_design_example": "/fred/oz016/Chayan/SNR_time_series_sample_files/example/default_snr_series_BNS_example_negative.hdf"
            
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
            
                "path_train_design_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_NSBH_3_det_design_0_sec_parameters_new_1.hdf",
                "path_train_design_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_NSBH_3_det_design_0_sec_parameters_new_2.hdf",
                "path_train_design_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_NSBH_3_det_design_0_sec_parameters_new_3.hdf",
                "path_train_design_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_NSBH_3_det_design_0_sec_parameters_new_4.hdf",
                "path_train_design_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_NSBH_3_det_design_0_sec_parameters_new_5.hdf",
            
            
                "path_test_Bayestar_post_merger_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_NSBH_10.hdf",
                "path_test_Bayestar_post_merger_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_NSBH_12.hdf",
                "path_test_Bayestar_post_merger_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_2.hdf",
                "path_test_Bayestar_post_merger_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_996.hdf",
                "path_test_Bayestar_post_merger_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_BBH_100.hdf",
                "path_test_Bayestar_post_merger_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_BBH_20.hdf",
                "path_test_Bayestar_post_merger_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_BBH_19.hdf",
            
                "path_train_real_noise_1":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_1/O2_noise_GW170817_NSBH_3_det_parameters_1.hdf",
                "path_train_real_noise_2":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_2/O2_noise_GW170817_NSBH_3_det_parameters_2.hdf",
                "path_train_real_noise_3":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_3/O2_noise_GW170817_NSBH_3_det_parameters_3.hdf",
                "path_train_real_noise_4":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_4/O2_noise_GW170817_NSBH_3_det_parameters_4.hdf",
                "path_train_real_noise_5":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/NSBH/bank_5/O2_noise_GW170817_NSBH_3_det_parameters_5.hdf",
            
                "path_test_GW190814": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_parameters_GW190814.hdf",
            
                "path_test_design_Bayestar_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_NSBH_3_det_design_0_sec_parameters_test_Bayestar_1.hdf",
            
                "path_test_Bayestar_post_merger": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_NSBH_10.hdf",
        
                "path_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_parameters_NSBH_test.hdf",
                "path_test_low_snr": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_snr-10to20_NSBH_test_parameters.hdf",
                "path_test_GW190917": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW190917_parameters_test_new_1.hdf"
            
                },
        
        "BBH": {
                "path_train": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_200k_injection_parameters",
                "path_train_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BBH_train_snr-10to20.hdf",
                
                "path_train_design_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BBH_3_det_design_0_sec_parameters_new_1.hdf",
                "path_train_design_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BBH_3_det_design_0_sec_parameters_new_2.hdf",
                "path_train_design_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BBH_3_det_design_0_sec_parameters_new_3.hdf",
                "path_train_design_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BBH_3_det_design_0_sec_parameters_new_4.hdf",
                "path_train_design_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BBH_3_det_design_0_sec_parameters_new_5.hdf",
            
                "path_train_real_noise_1":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_1/O2_noise_GW170817_BBH_3_det_parameters_1.hdf",
                "path_train_real_noise_2":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_2/O2_noise_GW170817_BBH_3_det_parameters_2.hdf",
                "path_train_real_noise_3":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_2/O2_noise_GW170817_BBH_3_det_parameters_3.hdf",
                "path_train_real_noise_4":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_3/O2_noise_GW170817_BBH_3_det_parameters_4.hdf",
                "path_train_real_noise_5":  "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/bank_4/O2_noise_GW170817_BBH_3_det_parameters_5.hdf",
            
                "path_train_real_noise_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_1/O2_noise_GW170817_BBH_3_det_parameters_unequal_masses_1.hdf",
                "path_train_real_noise_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_2/O2_noise_GW170817_BBH_3_det_parameters_unequal_masses_2.hdf",
                "path_train_real_noise_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_3/O2_noise_GW170817_BBH_3_det_parameters_unequal_masses_3.hdf",
                "path_train_real_noise_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/BBH/unequal_masses/bank_4/O2_noise_GW170817_BBH_3_det_parameters_unequal_masses_4.hdf",
                
                "path_train_O3_noise_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_1/O3_noise_GW170817_BBH_3_det_parameters_1.hdf",
                "path_train_O3_noise_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_2/O3_noise_GW170817_BBH_3_det_parameters_2.hdf", 
                "path_train_O3_noise_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_3/O3_noise_GW170817_BBH_3_det_parameters_3.hdf",
                "path_train_O3_noise_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/O3_noise/BBH/bank_4/O3_noise_GW170817_BBH_3_det_parameters_4.hdf", 
            
                "path_test_GW200224_222234": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_parameters_GW200224_222234.hdf",
                "path_test_GW190412": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_parameters_GW190412.hdf",
                "path_test_GW150914": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_parameters_GW150914.hdf",
                "path_test_GW170104": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_parameters_GW170104.hdf",
                "path_test_GW190521": "/fred/oz016/Chayan/SNR_time_series_sample_files/Real_events/real_event_parameters_GW190521.hdf",
                
            
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_BBH_snr-10to20_test_parameters.hdf",
                "path_test_design_Bayestar_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BBH_3_det_design_0_sec_parameters_test_Bayestar_1.hdf",
            
                "path_test_Bayestar_post_merger": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_BBH_100.hdf",
            
            
                "path_test_GW170729":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170729_parameters_test.hdf",
                "path_test_GW170809":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170809_parameters_test.hdf",
                "path_test_GW170814":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170814_parameters_test.hdf",
                "path_test_GW170818":"/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170818_parameters_test.hdf",
            
                },
        
        "BNS": {
                "path_train_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_1-6.hdf",
                "path_train_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BNS_train_7-24.hdf",
            
#                "path_train_real_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/default_GW170817_realnoise_parameters_1.hdf",
                "path_train_real_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/default_GW170817_realnoise_parameters_final_Kaya.hdf",
                "path_train_real_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/default_GW170817_realnoise_parameters_final_Pople.hdf",
            
                "path_train_2_det_high_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_snr-30to40_parameters_1.hdf",
                "path_train_2_det_high_SNR_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_snr-30to40_parameters_2.hdf",
                
            
                "path_train_low_snr_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_parameters_BNS_train_snr-10to20_2k_samples.hdf",
                "path_train_low_snr_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_parameters_BNS_train_snr-10to20_4k_samples.hdf",
                "path_train_low_snr_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_snr-10to20_train_OzSTAR_parameters.hdf",
            
                "path_train_5_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_5_sec_parameters_train.hdf",
                "path_train_10_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_10_sec_parameters_train.hdf",            
            
                "path_train_2_det_5_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_2_det_5_sec_parameters_1_train.hdf",
                "path_train_2_det_5_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_2_det_5_sec_parameters_2_train.hdf",
            
                "path_train_O4_PSD_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_parameters_1.hdf",
                "path_train_O4_PSD_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_parameters_2.hdf",
                "path_train_O4_PSD_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_parameters_3.hdf",
            
                "path_train_O4_PSD_5_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_5_sec_parameters_train_1.hdf",
                "path_train_O4_PSD_5_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_5_sec_parameters_train_2.hdf",
                "path_train_O4_PSD_5_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_5_sec_parameters_train_3.hdf",
                "path_train_O4_PSD_5_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_5_sec_parameters_train_4.hdf",
                "path_train_O4_PSD_5_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_O4_PSD_5_sec_parameters_train_5.hdf",
            
            
                # Design Sensitivity
                # 0 secs
                "path_train_design_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_1.hdf",
                "path_train_design_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_2.hdf",
                "path_train_design_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_3.hdf",
                "path_train_design_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_4.hdf",
                "path_train_design_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_5.hdf",
                "path_train_design_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_6.hdf",
                "path_train_design_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_7.hdf",
                "path_train_design_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_8.hdf",
                "path_train_design_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_9.hdf",
                "path_train_design_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_train_10.hdf",
            
                "path_train_design_high_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_high_snr_1.hdf",
                "path_train_design_high_SNR_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_high_snr_2.hdf",
            
                "path_train_O2_noise_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_1/O2_noise_GW170817_BNS_3_det_0_sec_parameters_1.hdf",
                "path_train_O2_noise_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_1/O2_noise_GW170817_BNS_3_det_0_sec_parameters_2.hdf",
                "path_train_O2_noise_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_2/O2_noise_GW170817_BNS_3_det_0_sec_parameters_3.hdf",
                "path_train_O2_noise_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_2/O2_noise_GW170817_BNS_3_det_0_sec_parameters_4.hdf",
                "path_train_O2_noise_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/GW170817_real_noise/bank_3/O2_noise_GW170817_BNS_3_det_0_sec_parameters_5.hdf",
                
                "path_test_design_high_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_test_high_SNR.hdf",
                "path_test_design_Bayestar_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_test_Bayestar.hdf",
            
                "path_test_Bayestar_post_merger": "/fred/oz016/Chayan/SNR_time_series_sample_files/Bayestar_post-merger_test/Bayestar_test_parameters_2.hdf",
            

                "path_test_design": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_0_sec_parameters_40k_test_new.hdf",
                "path_test_GW170817_0_secs": "/fred/oz016/Chayan/samplegen/output/default_test_GW170817_0_secs_parameters.hdf",
            
               "path_test_design_10_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_test_low_SNR_new.hdf",
                "path_test_GW170817_10_secs": "/fred/oz016/Chayan/samplegen/output/default_test_GW170817_10_secs_parameters.hdf",
            

                "path_test_design_15_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_test_low_SNR_new.hdf",
                "path_test_GW170817_15_secs": "/fred/oz016/Chayan/samplegen/output/default_test_GW170817_15_secs_parameters.hdf",
            
#                "path_test_design_30_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_8k_test_new.hdf",
                "path_test_design_30_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_test_low_SNR_new.hdf",
                "path_test_GW170817_30_secs": "/fred/oz016/Chayan/samplegen/output/default_test_GW170817_30_secs_parameters.hdf",
            
                "path_test_design_45_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_test_low_SNR_new.hdf",
#                "path_test_design_45_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_1/default_GW170817_BNS_3_det_45_sec_parameters_new_1.hdf",
                "path_test_GW170817_45_secs": "/fred/oz016/Chayan/samplegen/output/default_test_GW170817_45_secs_parameters.hdf",
            
                "path_test_design_58_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_test_low_SNR_new.hdf",
                "path_test_GW170817_58_secs": "/fred/oz016/Chayan/samplegen/output/default_test_GW170817_58_secs_parameters.hdf",
                            
                "path_test_design_example_0_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/example/default_premerger_test_parameters_0_sec.hdf",
            
                "path_test_design_example": "/fred/oz016/Chayan/SNR_time_series_sample_files/example/default_GW170817_3_det_BNS_example_negative_parameters.hdf",
            
                # 10 secs
                "path_train_design_10_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train.hdf",
                "path_train_design_10_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_1.hdf",
                "path_train_design_10_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_2.hdf",
                "path_train_design_10_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_3.hdf",
                "path_train_design_10_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_4.hdf",
                "path_train_design_10_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_5.hdf",
                "path_train_design_10_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_6.hdf",
                "path_train_design_10_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_7.hdf",
                "path_train_design_10_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_8.hdf",
                "path_train_design_10_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_9.hdf",
                "path_train_design_10_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_10.hdf",
                "path_train_design_10_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_11.hdf",
                "path_train_design_10_sec_13": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_12.hdf",
                "path_train_design_10_sec_14": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_train_13.hdf",
                "path_train_design_10_sec_15": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_14.hdf",
                "path_train_design_10_sec_16": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_15.hdf",
                "path_train_design_10_sec_17": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_16.hdf",
                "path_train_design_10_sec_18": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_17.hdf",
                "path_train_design_10_sec_19": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_18.hdf",
                "path_train_design_10_sec_20": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_10_sec_parameters_19.hdf",
                
            
                # 15 secs
                "path_train_design_15_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_1.hdf",
                "path_train_design_15_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_2.hdf",
                "path_train_design_15_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_3.hdf",
                "path_train_design_15_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_4.hdf",
                "path_train_design_15_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_5.hdf",
                "path_train_design_15_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_6.hdf",
                "path_train_design_15_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_7.hdf",
                "path_train_design_15_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_8.hdf",
                "path_train_design_15_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_train_9.hdf",
                "path_train_design_15_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_10.hdf",
                "path_train_design_15_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_11.hdf",
                "path_train_design_15_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_15_sec_parameters_12.hdf",
            
                # 30 secs
                "path_train_design_30_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_1.hdf",
                "path_train_design_30_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_2.hdf",
                "path_train_design_30_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_3.hdf",
                "path_train_design_30_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_4.hdf",
                "path_train_design_30_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_5.hdf",
                "path_train_design_30_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_6.hdf",
                "path_train_design_30_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_7.hdf",
                "path_train_design_30_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_8.hdf",
                "path_train_design_30_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_9.hdf",
                "path_train_design_30_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_10.hdf",
                "path_train_design_30_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_11.hdf",
                "path_train_design_30_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_30_sec_parameters_train_12.hdf",
            
                # 45 secs
                "path_train_design_45_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_1.hdf",
                "path_train_design_45_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_2.hdf",
                "path_train_design_45_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_3.hdf",
                "path_train_design_45_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_4.hdf",
                "path_train_design_45_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_5.hdf",
            
            
                "path_train_design_45_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_6.hdf",
                "path_train_design_45_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_7.hdf",
                "path_train_design_45_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_8.hdf",
                "path_train_design_45_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_9.hdf",
                "path_train_design_45_sec_10": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_10.hdf",
                "path_train_design_45_sec_11": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_45_sec_parameters_train_11.hdf",
            
            
                "path_train_design_45_sec_12": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_1/default_GW170817_BNS_3_det_45_sec_parameters_new_1.hdf",
                "path_train_design_45_sec_13": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_2/default_GW170817_BNS_3_det_45_sec_parameters_new_2.hdf",
                "path_train_design_45_sec_14": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_3/default_GW170817_BNS_3_det_45_sec_parameters_new_3.hdf",
                "path_train_design_45_sec_15": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_4/default_GW170817_BNS_3_det_45_sec_parameters_new_4.hdf",
                "path_train_design_45_sec_16": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/45_secs_new/bank_4/default_GW170817_BNS_3_det_45_sec_parameters_new_5.hdf",
            
                # 58 secs
                "path_train_design_58_sec_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_1.hdf",
                "path_train_design_58_sec_2": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_2.hdf",
                "path_train_design_58_sec_3": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_3.hdf",
                "path_train_design_58_sec_4": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_4.hdf",
                "path_train_design_58_sec_5": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_5.hdf",
                "path_train_design_58_sec_6": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_6.hdf",
                "path_train_design_58_sec_7": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_7.hdf",
                "path_train_design_58_sec_8": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_8.hdf",
                "path_train_design_58_sec_9": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_3_det_design_58_sec_parameters_train_9.hdf",                
            
                "path_train_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_BNS_2_det_snr-10to20_low_mass_parameters.hdf",
                "path_train_2_det_low_SNR_1": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_parameters_BNS_train_snr-10to20_new.hdf",                            
            
                "path_test": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_BNS_test.hdf",
                "path_test_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_parameters_test_snr_10-20_BNS_test.hdf",
                "path_test_GW170817": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_Gaussian_noise_parameters_test.hdf",
            
                "path_test_2_det_high_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/default_GW170817_2_det_snr-30to40_parameters_test.hdf",
                
                "path_test_2_det_low_SNR": "/fred/oz016/Chayan/SNR_time_series_sample_files/Kaya_data/default_GW170817_2_det_snr_10-20_BNS_test_parameters.hdf",
                            
                "path_test_3_det_5_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_aLIGO_PSD_5_sec_test_parameters.hdf",
                "path_test_3_det_10_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_aLIGO_PSD_10_sec_test_parameters.hdf",
            
                "path_test_2_det_0_secs": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_BNS_2_det_0_sec_parameters.hdf",
            
                "path_test_O4_PSD_0_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_O4_PSD_0_sec_parameters_test.hdf",
                "path_test_O4_PSD_5_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_O4_PSD_5_sec_parameters_test.hdf",
                "path_test_O4_PSD_10_sec": "/fred/oz016/Chayan/SNR_time_series_sample_files/Negative_latency/default_GW170817_O4_PSD_10_sec_parameters_test.hdf"
                
                },   
    },
    "train": {
            "network": "ResNet-34",
            "dataset": "BBH",
            "train_real": False,
            "test_real": False,
            "PSD": 'design', # 'O4'/'aLIGO/design'
            "train_negative_latency": False,
            "train_negative_latency_seconds": '0', 
            "test_negative_latency": False,
            "test_negative_latency_seconds": '0', 
            "snr_range_train": 'low',
            "snr_range_test": 'low',
            "num_train": 546000, 
            "num_test": 40000,
            "min_snr": 8,
            "n_samples": 410,
            "batch_size": 4000,
            "output_filename": 'Adaptive_NSIDE/Negative_latency/Injection_run_BNS_3_det_O4_PSD_test.hdf',
            "checkpoint_restore": False,
            "num_detectors": 3,
                "epochs": 5,
            "validation_split": 0.05,
            "optimizer": {
                "type": "adam"
            },
    },
    "model": { 
        
            "num_bijectors": 5,
            "MAF_hidden_units": [256, 256, 256, 256, 256],
        
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
                        "filters_real":32, 
                        "filters_imag":32,
                        "kernel_size" : (2,7),
                        "strides": 2,
                        "pool_size": (1,3),
                        "prev_filters_real":32,
                        "prev_filters_imag":32 
            },    
            "ResNet_50": {
                        "filters_real":64, 
                        "filters_imag":64, 
                        "kernel_size" : 7,
                        "strides": 2,
                        "pool_size": 3,
                        "prev_filters_real":64, 
                        "prev_filters_imag":64 
            
            },
            "ResNet_34_2_det": {
                        "filters_real": 32,
                        "filters_imag": 32, 
                        "kernel_size" : 7,
                        "strides": 2,
                        "pool_size": 3,
                        "prev_filters_real": 32,
                        "prev_filters_imag": 32 
            
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
