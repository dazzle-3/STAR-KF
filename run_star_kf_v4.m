function outputs = run_star_kf_v4(input_dir, output_dir, cfg)
%RUN_STAR_KF_V4 Unified STAR-KF v4 two-stage joint-flow entry point.

root_dir = fileparts(mfilename('fullpath'));
addpath(root_dir, fullfile(root_dir, 'modules'), ...
    fullfile(root_dir, 'utils'), fullfile(root_dir, 'evaluation'));
run_tic = tic;

if nargin < 3 || isempty(cfg)
    cfg = star_kf_v4_config();
end
if nargin < 1
    input_dir = '';
end
if nargin < 2
    output_dir = '';
end

[input_dir, input_source] = resolve_input_dir( ...
    input_dir, root_dir, cfg);
output_dir = resolve_output_dir(output_dir, root_dir, cfg);
assert_safe_output_directory(output_dir, root_dir);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

log_progress('RUN', ...
    'Starting STAR-KF v4 two-stage joint flow: input_dir=%s, output_dir=%s', ...
    input_dir, output_dir);
log_progress('RUN', ...
    'Configuration: profile=%s, random_model=%s, date=%s', ...
    cfg.runtime.profile, cfg.runtime.random_model_variant, ...
    cfg.experiment.configuration_date);
log_progress('RUN', ...
    ['Spatial model=%s, candidate_estimator=%s, block=%.3f km, ', ...
     'nominal_step=%.3f km'], ...
    cfg.kernel.model_variant, cfg.spatial_estimator.variant, ...
    cfg.kriging.block_size_km, ...
    cfg.kriging.block_size_km * (1 - cfg.kriging.overlap_ratio));
log_progress('RUN', 'Input source: %s', input_source);

t_load = tic;
data = load_experiment_data(input_dir, cfg);
log_progress('DATA', ...
    'Loaded %d points, %d IFGs, %d SLCs in %.2f s', ...
    data.n_pts, data.n_ifg, data.n_slc, toc(t_load));

result = joint_inversion(data, cfg);
validate_deformation_output_contract(result);

log_progress('EVAL', 'Computing quantitative metrics');
metrics = evaluate_quantitative_metrics(data, result, output_dir);

result_path = fullfile(output_dir, 'joint_inversion_results.mat');
log_progress('IO', 'Saving outputs to %s', result_path);
save(result_path, 'data', 'result', 'metrics', 'cfg', '-v7.3');
write_run_manifest( ...
    output_dir, input_dir, input_source, result_path, ...
    cfg, data, result, metrics);
write_reproducibility_manifest(root_dir,input_dir,output_dir,cfg,result);
cleanup_parallel_pool();

outputs = struct( ...
    'data', data, ...
    'result', result, ...
    'metrics', metrics, ...
    'cfg', cfg, ...
    'input_dir', input_dir, ...
    'output_dir', output_dir, ...
    'result_path', result_path);

log_progress('RUN', 'STAR-KF v4 completed in %.2f s', toc(run_tic));
fprintf('STAR-KF v4 two-stage joint flow finished\n');
fprintf('Profile: %s\n', cfg.runtime.profile);
fprintf('Points: %d, IFGs: %d, SLCs: %d\n', ...
    data.n_pts, data.n_ifg, data.n_slc);
fprintf('Input dir: %s\n', input_dir);
fprintf('Output dir: %s\n', output_dir);
end

function validate_deformation_output_contract(result)
required = {'deformation_joint_final', ...
    'timeseries_atm_corrected', 'turbulence_final', ...
    'residual_final', 'R_final_mm2', 'P_joint_final_mm2', ...
    'iterations', 'phi_seq_filter_mm', ...
    'phi_seq_corrected_mm', 'phi_turb_final_rad', ...
    'phi_turb_seq_mm', 'phi_def_filter_ifg_rad', ...
    'phi_def_corrected_ifg_rad'};
for i = 1:numel(required)
    if ~isfield(result, required{i})
        error('Missing required output: result.%s', required{i});
    end
end

forbidden = {'phi_seq_iterback_mm', 'iterback_history', ...
    'phi_seq_recovered_mm', 'phi_def_recovered_ifg_rad', ...
    'phi_seq_final_mm'};
for i = 1:numel(forbidden)
    if isfield(result, forbidden{i})
        error('Deprecated output must not be saved: result.%s', ...
            forbidden{i});
    end
end

if ~isequaln(result.phi_seq_corrected_mm, ...
        result.deformation_joint_final)
    error(['result.phi_seq_corrected_mm must be a compatibility alias ', ...
        'of result.deformation_joint_final.']);
end
if ~isequaln(result.phi_turb_final_rad, result.turbulence_final)
    error(['result.phi_turb_final_rad must be a compatibility alias ', ...
        'of result.turbulence_final.']);
end

contract = result.output_contract;
if contract.post_kriging_deformation_recovery || ...
        ~strcmp(contract.formal_flow, 'KF1_KF2_then_spatial_modeling')
    error('The two-stage joint output contract is invalid.');
end
for q = 1:numel(result.iterations)
    if ~result.iterations{q}.fresh_spatial_update
        error('Every saved formal iteration must contain a fresh spatial update.');
    end
end
end

function [input_dir, input_source] = resolve_input_dir( ...
    input_dir, root_dir, cfg)
if ~isempty(input_dir)
    if ~has_required_inputs(input_dir, cfg)
        error('Input directory is missing %s or %s: %s', ...
            cfg.input_file, cfg.kf_input_file, input_dir);
    end
    input_source = 'user_supplied';
    return;
end

candidates = {
    fullfile(root_dir, cfg.experiment.recommended_input_dir)
    fullfile(root_dir, 'data')
    };
for i = 1:numel(candidates)
    if has_required_inputs(candidates{i}, cfg)
        input_dir = candidates{i};
        input_source = 'configured_default';
        return;
    end
end
error(['An input directory containing %s and %s is required. ', ...
    'Call run_star_kf_v4(input_dir).'], ...
    cfg.input_file, cfg.kf_input_file);
end

function output_dir = resolve_output_dir(output_dir, root_dir, cfg)
if isempty(output_dir)
    output_dir = fullfile( ...
        root_dir, cfg.experiment.output_dir_name);
end
end

function tf = has_required_inputs(input_dir, cfg)
tf = isfolder(input_dir) && ...
    isfile(fullfile(input_dir, cfg.input_file)) && ...
    isfile(fullfile(input_dir, cfg.kf_input_file));
end

function assert_safe_output_directory(output_dir, root_dir)
protected_names = {
    'outputs_filter_tuned_taper_balanced_full_leakrec'
    'outputs_filter_tuned_taper_balanced_full'
    'outputs_filter_tuned_taper_balanced_full_defguard_v1'
    'outputs_filter_tuned_taper_balanced_full_defguard_v2_adaptive'
    };
resolved_output = canonical_path(output_dir);
for i = 1:numel(protected_names)
    protected = canonical_path(fullfile(root_dir, protected_names{i}));
    if strcmpi(resolved_output, protected)
        error(['Refusing to overwrite completed legacy result ', ...
            'directory: %s.'], output_dir);
    end
end
end

function path_out = canonical_path(path_in)
path_in = char(path_in);
if isfolder(path_in)
    [ok, attributes] = fileattrib(path_in);
    if ok
        path_out = attributes.Name;
    else
        path_out = path_in;
    end
elseif is_absolute_path(path_in)
    path_out = path_in;
else
    path_out = fullfile(pwd, path_in);
end
path_out = strrep(path_out, '/', filesep);
while numel(path_out) > 3 && ...
        (path_out(end) == '\' || path_out(end) == '/')
    path_out(end) = [];
end
end

function tf = is_absolute_path(path_in)
tf = ~isempty(regexp(path_in, '^[A-Za-z]:[\\/]', 'once')) || ...
    startsWith(path_in, '\\');
end

function write_run_manifest( ...
    output_dir, input_dir, input_source, result_path, ...
    cfg, data, result, metrics)
manifest_path = fullfile(output_dir, 'experiment_setup_summary.md');
lines = {
    '# STAR-KF v4 Two-Stage Joint Experiment'
    ''
    sprintf('- Profile: %s', cfg.runtime.profile)
    sprintf('- Configuration date: %s', ...
        cfg.experiment.configuration_date)
    sprintf('- Input dir: %s', input_dir)
    sprintf('- Input source: %s', input_source)
    sprintf('- Output dir: %s', output_dir)
    sprintf('- Result file: %s', result_path)
    sprintf('- Epoch unit: %s', cfg.units.internal_epoch)
    sprintf('- IFG unit: %s', cfg.units.internal_ifg)
    sprintf('- scale_rad2mm: %.9f', cfg.scale_rad2mm)
    sprintf('- Spatial covariance model: %s', ...
        cfg.kernel.model_variant)
    sprintf('- Turbulence candidate estimator: %s', ...
        cfg.spatial_estimator.variant)
    sprintf('- Kriging block size: %.3f km', ...
        cfg.kriging.block_size_km)
    sprintf('- Kriging nominal step: %.3f km', ...
        cfg.kriging.block_size_km * (1 - cfg.kriging.overlap_ratio))
    ''
    'Formal flow:'
    '- G(phi_original - tau_prev; R_current, random_model)'
    '- KF1 fixed A1 -> transformed full-trajectory KF2 -> three-component acceptance -> D_joint'
    '- covariance -> VCE -> absolute Kriging candidate'
    '- controlled turbulence update -> E_final -> R_next -> convergence'
    '- fixed tau_final/R_final final KF1+KF2 consistency pass'
    sprintf('- KF2 q_acc: %.6g', cfg.kalman_kf2.q_acc)
    '- KF2 uses L1 SigmaY L1'' and L1 Kd L1'', ordered reduced Cholesky whitening'
    sprintf('- Turbulence beta: %.3f', cfg.turbulence_update.beta)
    sprintf('- Termination metric: %s < %.4g', ...
        result.iterations{end}.convergence.termination_metric, ...
        result.iterations{end}.convergence.thresholds.C)
    sprintf('- Consecutive fresh updates required: %d', ...
        cfg.convergence.consecutive_converged_iters)
    ''
    'Output contract:'
    '- deformation_joint_final: dynamic-constrained joint deformation (LOS mm)'
    '- timeseries_atm_corrected: network-only corrected time series (LOS mm)'
    '- turbulence_final: absolute final IFG turbulence field (rad)'
    '- residual_final: complete final IFG residual (rad)'
    sprintf('- post-Kriging deformation recovery: %d', ...
        result.output_contract.post_kriging_deformation_recovery)
    ''
    'Dataset:'
    sprintf('- Points: %d', data.n_pts)
    sprintf('- IFGs: %d', data.n_ifg)
    sprintf('- SLCs: %d', data.n_slc)
    ''
    'Diagnostics:'
    sprintf('- Model-fit RMSE: %.4f mm', metrics.model_fit_rmse_mm)
    sprintf('- KF1/joint RMSE: %.4f / %.4f mm', ...
        metrics.branch_model_fit_rmse_mm.filter, ...
        metrics.branch_model_fit_rmse_mm.corrected)
    sprintf('- Completed outer iterations: %d', ...
        numel(result.iterations))
    sprintf('- Termination reason: %s',result.termination_reason)
    sprintf('- Criterion fulfilled: %d; iteration limit reached: %d', ...
        result.criterion_fulfilled,result.iteration_limit_reached)
    sprintf('- Diagnostic continuation: %d; first criterion round (0=never): %d', ...
        result.diagnostic_continue_after_convergence,result.criterion_first_satisfied_iteration)
    sprintf('- Final conditional d_final diagnostic: %.8g',result.finalization.d_final)
    sprintf('- Fresh spatial updates: %d', ...
        result.runtime_profile.fresh_spatial_updates)
    };

fid = fopen(manifest_path, 'w', 'n', 'UTF-8');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
for i = 1:numel(lines)
    fprintf(fid, '%s\n', lines{i});
end
end

function cleanup_parallel_pool()
pool = gcp('nocreate');
if ~isempty(pool)
    log_progress('RUN', 'Closing parallel pool');
    delete(pool);
end
end
