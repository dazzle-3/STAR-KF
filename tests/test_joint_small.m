function tests = test_joint_small
tests = functiontests(localfunctions);
end

function setupOnce(~)
project = fileparts(fileparts(mfilename('fullpath')));
addpath(project, fullfile(project, 'modules'), ...
    fullfile(project, 'utils'), fullfile(project, 'evaluation'), ...
    fullfile(project, 'tests'));
end

function testMixedFormalFlow(testCase)
[data, ~] = make_small_synthetic_case('mixed', 11);
cfg = small_test_config(2);
result = joint_inversion(data, cfg);
verifySize(testCase, result.deformation_joint_final, ...
    [data.n_pts, data.n_slc]);
verifySize(testCase, result.turbulence_final, ...
    [data.n_pts, data.n_ifg]);
expected_residual = data.phi_mat - ...
    sequence_to_ifg(result.deformation_joint_final, ...
        data.ifg_pairs, 1 / data.scale_rad2mm) - ...
    result.turbulence_final;
verifyEqual(testCase, result.residual_final, expected_residual, ...
    'AbsTol', 1e-10);
verifyTrue(testCase, all(isfinite(result.R_final_mm2)));
verifyGreaterThanOrEqual(testCase, min(result.R_final_mm2), ...
    cfg.uncertainty.r_floor_mm2);
verifyLessThanOrEqual(testCase, max(result.R_final_mm2), ...
    cfg.uncertainty.r_ceiling_mm2);
verifyFalse(testCase, result.output_contract.post_kriging_deformation_recovery);
for q = 1:numel(result.iterations)
    iteration = result.iterations{q};
    verifyTrue(testCase, iteration.fresh_spatial_update);
    verifySize(testCase, iteration.W2, [data.n_pts, 1]);
    verifyTrue(testCase, all(isfinite(iteration.R_next)));
end
end

