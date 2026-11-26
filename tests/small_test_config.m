function cfg=small_test_config(~)
% Runtime support for the compact fixture; manuscript constants are unchanged.
cfg=star_kf_v4_config('initialprior_small_validation','all_random_model');
cfg.i_master=4;
cfg.kalman.parallel_enable=false; cfg.kalman_kf2.parallel_enable=false;
cfg.kernel.n_sample_cov=64; cfg.kernel.shared_sample_size=64;
cfg.kernel.min_valid_pts=12; cfg.kernel.parallel_enable=false;
cfg.operator_output.retain_all=true;
end
