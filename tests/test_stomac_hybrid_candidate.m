function tests=test_stomac_hybrid_candidate
tests=functiontests(localfunctions);
end
function testHybridCannotMixMeanAndVarianceInFormalFlow(testCase)
[data,~]=make_small_synthetic_case('mixed',20260808); cfg=small_test_config();
cfg.spatial_estimator.variant='stomac_epoch_evd_blup';
verifyError(testCase,@()joint_inversion(data,cfg),'STAR_KF:FormalSpatial');
cfg=star_kf_v4_config(); verifyEqual(testCase,cfg.spatial_estimator.variant,'kriging');
end
