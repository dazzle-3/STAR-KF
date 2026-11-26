function tests=test_spatial_model_variants
tests=functiontests(localfunctions);
end
function testFormalMethodRejectsHistoricalVariants(testCase)
[data,~]=make_small_synthetic_case('mixed',20260807);
variants={'single_exponential_isotropic_stationary','dual_matern_isotropic_stationary','dual_matern_anisotropic_nonstationary'};
for i=1:numel(variants)
    cfg=small_test_config(); cfg.kernel.model_variant=variants{i};
    verifyError(testCase,@()joint_inversion(data,cfg),'STAR_KF:FormalSpatial');
end
cfg=star_kf_v4_config();
verifyEqual(testCase,cfg.kernel.model_variant,'dual_matern_anisotropic_stationary');
end
