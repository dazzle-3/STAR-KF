function tests = test_performance_equivalence
tests = functiontests(localfunctions);
end

function testKalmanAndCovarianceAreDeterministic(testCase)
rng(9107);
cfg = star_kf_v4_config([], 'all_random_model');
t = 0:12:360;
z = 0.03*t + 0.7*sin(2*pi*t/365.25) + ...
    0.08*randn(1, numel(t));
rv = linspace(5, 9, numel(t));
models = {'cv', 'ca', 'seasonal'};
for j = 1:numel(models)
    first = kalman_filter(z, t, rv, cfg.kalman, models{j}, 16);
    second = kalman_filter(z, t, rv, cfg.kalman, models{j}, 16);
    verifyEqual(testCase, second.sequence_mm, first.sequence_mm, ...
        'AbsTol', 0);
    verifyEqual(testCase, second.state_var_mm2, first.state_var_mm2, ...
        'AbsTol', 0);
    first = kalman_filter(z - 0.02*t, t, rv, ...
        cfg.kalman_kf2, models{j}, 16);
    second = kalman_filter(z - 0.02*t, t, rv, ...
        cfg.kalman_kf2, models{j}, 16);
    verifyEqual(testCase, second.sequence_mm, first.sequence_mm, ...
        'AbsTol', 0);
    verifyEqual(testCase, second.state_var_mm2, first.state_var_mm2, ...
        'AbsTol', 0);
end

coord = [repelem((0:29)', 30), repmat((0:29)', 30, 1)] * 0.2;
resid = 0.12 * randn(900, 7);
params = cfg.kernel;
params.shared_sample_size = 800;
params.n_sample_cov = 800;
params.min_valid_pts = 100;
params.parallel_enable = false;
[first, cache] = stomac_covariance( ...
    coord, resid, params, [], true(900, 1));
[second, cache_reused] = stomac_covariance( ...
    coord, resid, params, cache, true(900, 1));
fields = {'c_short_ifg', 'c_long_ifg', 'sigma2_ifg', ...
    'range_x_ifg', 'range_y_ifg', 'range_long_ifg', ...
    'anisotropy_ratio'};
for j = 1:numel(fields)
    verifyEqual(testCase, second.(fields{j}), ...
        first.(fields{j}), 'AbsTol', 0);
end
verifyEqual(testCase, cache.cache_format, 'compact_pairs_v1');
verifyEqual(testCase, cache_reused.cache_format, 'compact_pairs_v1');
end

function testParallelKalmanAndCovarianceAreExactlySerial(testCase)
rng(223);
cfg = star_kf_v4_config( ...
    [], 'all_random_model');
n = 600;
t = 0:12:360;
Z = 0.03 * repmat(t, n, 1) + 0.2 * randn(n, numel(t));
rv = linspace(5, 9, numel(t));
serial_params = cfg.kalman;
serial_params.parallel_enable = false;
serial = multi_model_filter(Z, t, serial_params, rv, 16);
parallel_params = cfg.kalman;
parallel_params.parallel_enable = true;
parallel_params.parallel_pool_type = 'threads';
parallel_params.parallel_chunk_size = 50000;
parallel = multi_model_filter(Z, t, parallel_params, rv, 16);
verifyTrue(testCase, parallel.runtime_stats.parallel_used);
verifyEqual(testCase, parallel.phi_seq_mm, serial.phi_seq_mm, ...
    'AbsTol', 0);
verifyEqual(testCase, parallel.phi_seq_var_mm2, ...
    serial.phi_seq_var_mm2, 'AbsTol', 0);

coord = [repelem((0:29)', 30), repmat((0:29)', 30, 1)] * 0.2;
resid = 0.12 * randn(900, 20);
params = cfg.kernel;
params.shared_sample_size = 800;
params.n_sample_cov = 800;
params.min_valid_pts = 100;
params.parallel_enable = false;
[serial_cov, cache] = stomac_covariance( ...
    coord, resid, params, [], true(900, 1));
params.parallel_enable = true;
params.parallel_min_ifgs = 8;
parallel_cov = stomac_covariance( ...
    coord, resid, params, cache, true(900, 1));
verifyTrue(testCase, parallel_cov.quality.parallel_used);
fields = {'c_short_ifg', 'c_long_ifg', 'sigma2_ifg', ...
    'range_x_ifg', 'range_y_ifg', 'range_long_ifg'};
for j = 1:numel(fields)
    verifyEqual(testCase, parallel_cov.(fields{j}), ...
        serial_cov.(fields{j}), 'AbsTol', 0);
end
end

function testKrigingGeometryCacheAndJointOutputContracts(testCase)
[data, ~] = make_small_synthetic_case('mixed', 92);
cfg = small_test_config(2);
[covariance, ~] = stomac_covariance(data.coord_km, data.phi_mat, ...
    cfg.kernel, [], true(data.n_pts, 1));
vce = vce_estimation(covariance, data.ifg_pairs, data.n_slc, cfg.vce);
noise = build_heteroscedastic_noise_model( ...
    data.coord_km, data.phi_mat, cfg.optimization);
[first, cache] = kriging_interpolation(data.coord_km, data.phi_mat, ...
    covariance, vce, data.ifg_pairs, cfg.kriging, cfg.kernel, ...
    noise, true(data.n_pts, 1), []);
[second, ~] = kriging_interpolation(data.coord_km, data.phi_mat, ...
    covariance, vce, data.ifg_pairs, cfg.kriging, cfg.kernel, ...
    noise, true(data.n_pts, 1), cache);
verifyEqual(testCase, second.phi_turb, first.phi_turb, 'AbsTol', 0);
verifyEqual(testCase, second.phi_turb_var, ...
    first.phi_turb_var, 'AbsTol', 0);
verifyFalse(testCase, first.quality.geometry_cache_reused);
verifyTrue(testCase, second.quality.geometry_cache_reused);

actual = joint_inversion(data, cfg);
verifySize(testCase, actual.deformation_joint_final_mm, ...
    [data.n_pts, data.n_slc]);
verifySize(testCase, actual.turbulence_final_rad, ...
    [data.n_pts, data.n_ifg]);
verifyTrue(testCase, all(isfinite( ...
    actual.deformation_joint_final_mm(:))));
verifyTrue(testCase, all(isfinite(actual.turbulence_final_rad(:))));
verifyTrue(testCase, all(isfinite(actual.R_final_mm2(:))));
verifyEqual(testCase, actual.deformation_joint_final_mm(:, data.i_master), ...
    zeros(data.n_pts, 1), 'AbsTol', 1e-12);
end
