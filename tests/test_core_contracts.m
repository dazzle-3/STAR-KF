function tests = test_core_contracts
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
project = fileparts(fileparts(mfilename('fullpath')));
addpath(project, fullfile(project, 'modules'), ...
    fullfile(project, 'utils'), fullfile(project, 'tests'));
testCase.TestData.project = project;
end

function testIfgEpochRoundTrip(testCase)
[data, ~] = make_small_synthetic_case('pure_deformation', 1);
sequence = data.truth.phi_def_mm;
ifg = sequence_to_ifg(sequence, data.ifg_pairs, 1);
recovered = recover_sequence_phase( ...
    ifg, data.n_slc, data.ifg_pairs, data.i_master);
verifyLessThan(testCase, max(abs(recovered(:) - sequence(:))), 1e-9);
verifyEqual(testCase, recovered(:, data.i_master), ...
    zeros(data.n_pts, 1), 'AbsTol', 1e-12);
end

function testIfgVariancePropagation(testCase)
[data, ~] = make_small_synthetic_case('mixed', 2);
ifg_var = linspace(0.1, 2.0, data.n_ifg);
[epoch_var, diagnostics] = ifg_variance_to_epoch_variance( ...
    ifg_var, data.ifg_pairs, data.n_slc, data.i_master, mapping_settings());
verifySize(testCase, epoch_var, [1, data.n_slc]);
verifyTrue(testCase, all(isfinite(epoch_var)));
verifyGreaterThanOrEqual(testCase, min(epoch_var), 0);
verifyEqual(testCase, epoch_var(data.i_master), 0, 'AbsTol', 1e-12);
verifyGreaterThanOrEqual(testCase, diagnostics.rank, data.n_slc - 1);
end

function testSparsePositiveIfgVarianceDoesNotDisconnectNetwork(testCase)
[data, ~] = make_small_synthetic_case('mixed', 21);
ifg_var = zeros(1, data.n_ifg);
ifg_var(1:min(5, data.n_ifg)) = linspace(0.5, 4.0, min(5, data.n_ifg));
[epoch_var, diagnostics] = ifg_variance_to_epoch_variance( ...
    ifg_var, data.ifg_pairs, data.n_slc, data.i_master, mapping_settings());
[scaled_var, ~] = ifg_variance_to_epoch_variance( ...
    2 .* ifg_var, data.ifg_pairs, data.n_slc, data.i_master, mapping_settings());
verifyTrue(testCase, all(isfinite(epoch_var)));
verifyEqual(testCase, epoch_var(data.i_master), 0, 'AbsTol', 1e-12);
verifyLessThan(testCase, max(epoch_var), 10 .* max(ifg_var));
verifyEqual(testCase, scaled_var, 2 .* epoch_var, 'RelTol', 1e-8, ...
    'AbsTol', 1e-10);
verifyGreaterThanOrEqual(testCase, diagnostics.rank, data.n_slc - 1);
verifyGreaterThan(testCase, diagnostics.zero_variance_count, 0);
verifyEqual(testCase, diagnostics.method, 'unloaded_linear_mapper_covariance');
end

function testKalmanReturnsReferenceRelativeVariance(testCase)
cfg = star_kf_v4_config();
t = 0:12:120;
z = 0.3 .* (t - t(6));
R = 16 .* ones(size(t));
result = kalman_filter(z, t, R, cfg.kalman, 'cv', 6);
verifyEqual(testCase, result.state_var_mm2(6), 0, 'AbsTol', 1e-12);
verifyTrue(testCase, all(isfinite(result.state_var_mm2)));
verifyTrue(testCase, all(result.state_var_mm2 >= 0));
verifyGreaterThan(testCase, median(result.absolute_state_var_mm2), 0);
end


function testEpochTurbulenceAndJointCovariance(testCase)
[data,~]=make_small_synthetic_case('mixed',3); cfg=star_kf_v4_config();
[B,~,~,~]=network_incidence(data.ifg_pairs,data.n_slc,data.i_master);
[~,map]=ifg_variance_to_epoch_variance(ones(1,data.n_ifg),data.ifg_pairs,data.n_slc,data.i_master,cfg.variance_mapping);
previous=zeros(data.n_pts,data.n_slc); candidate=.01*sin(reshape(1:numel(data.phi_mat),size(data.phi_mat)));
[tau,diagnostics,a]=controlled_turbulence_update(previous,candidate,zeros(size(candidate)),cfg.turbulence_update,data,map.mapper_phase,true(size(candidate)));
verifyEqual(testCase,tau,a*B','AbsTol',0);
verifyEqual(testCase,diagnostics.clipped_fraction,0,'AbsTol',0);
T=eye(data.n_slc); T(:,data.i_master)=T(:,data.i_master)-1;
S=T*T'; A1=.2*T; F2=.3*T;
[P,CD]=build_joint_deformation_uncertainty(A1,F2,S,.4);
verifyEqual(testCase,P,diag(CD)','AbsTol',1e-12);
verifyEqual(testCase,CD,(A1+.4*F2)*S*(A1+.4*F2)','AbsTol',1e-12);
end

function testCurrentOptimizedParametersAreCanonical(testCase)
current = star_kf_v4_config([], 'all_random_model');
compatibility_label = star_kf_v4_config( ...
    'synthetic_generator_tuned_experiment', 'all_random_model');
verifyEqual(testCase, current.i_master, 16);
verifyEqual(testCase,current.joint.n_outer_iters,5);
verifyFalse(testCase,current.diagnostics.continue_after_convergence);
verifyEqual(testCase, current.kalman.q_acc, 2.0e-5, 'AbsTol', 1e-14);
verifyEqual(testCase, current.kalman.R_floor, 3.0, 'AbsTol', 1e-14);
verifyFalse(testCase, current.kalman.adaptive_Q);
verifyEqual(testCase, current.kalman.imm_self_prob, 0.98, ...
    'AbsTol', 1e-14);
verifyEqual(testCase, current.kalman.model_prior, [0.40, 0.30, 0.30], ...
    'AbsTol', 1e-14);
verifyTrue(testCase, current.kalman.parallel_enable);
verifyEqual(testCase, current.kalman.parallel_pool_type, 'threads');
verifyEqual(testCase, current.kalman.parallel_chunk_size, 50000);
verifyTrue(testCase, current.kernel.parallel_enable);
verifyTrue(testCase, current.experiment.experimental);
verifyTrue(testCase, current.experiment.synthetic_truth_tuned);
verifyEqual(testCase, current.experiment.reporting_scope, ...
    'simulation_experiment_only_not_for_real_data_reporting');
verifyEqual(testCase, current.convergence.joint_state_tol, 0.01, ...
    'AbsTol', 1e-14);
verifyFalse(testCase, isfield(current.convergence, 'deformation_tol'));
verifyFalse(testCase, isfield(current.convergence, 'turbulence_tol'));
verifyFalse(testCase, ...
    isfield(current.convergence, 'observation_variance_tol'));

% The profile argument is now metadata only. It cannot select the deleted
% historical formal parameter set or otherwise change algorithm settings.
verifyEqual(testCase, compatibility_label.i_master, current.i_master);
verifyEqual(testCase, compatibility_label.kalman, current.kalman);
verifyEqual(testCase, compatibility_label.kalman_kf2, current.kalman_kf2);
verifyEqual(testCase, compatibility_label.kf2_acceptance, ...
    current.kf2_acceptance);
end

function settings=mapping_settings()
cfg=star_kf_v4_config(); settings=cfg.variance_mapping;
end
