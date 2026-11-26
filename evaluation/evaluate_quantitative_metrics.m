function metrics = evaluate_quantitative_metrics(data, result, output_dir)
metrics = compute_reference_free_metrics(data, result);
metrics.available = false;
metrics.has_truth = isfield(data, 'truth') && isfield(data.truth, 'phi_def_mm') && isfield(data.truth, 'phi_turb_mm');

if metrics.has_truth
    metrics = add_truth_based_metrics(metrics, data, result);
else
    metrics.message = 'Ground truth is unavailable; reporting reference-free diagnostics only.';
end

if nargin >= 3 && ~isempty(output_dir)
    write_metrics_summary(metrics, fullfile(output_dir, 'quantitative_metrics_summary.md'));
end
end

function metrics = compute_reference_free_metrics(data, result)
obs_ifg_mm = data.phi_mat * data.scale_rad2mm;
filter_ifg_mm = result.phi_def_filter_ifg_rad * data.scale_rad2mm;
corrected_ifg_mm = result.phi_def_corrected_ifg_rad * data.scale_rad2mm;
turb_ifg_mm = result.phi_turb_final_rad * data.scale_rad2mm;
model_ifg_mm = corrected_ifg_mm + turb_ifg_mm;
final_resid_ifg_mm = obs_ifg_mm - model_ifg_mm;
filter_resid_ifg_mm = obs_ifg_mm - (filter_ifg_mm + turb_ifg_mm);
corrected_resid_ifg_mm = obs_ifg_mm - (corrected_ifg_mm + turb_ifg_mm);
raw_def_misfit_mm = obs_ifg_mm - corrected_ifg_mm;

obs_var = variance_all(obs_ifg_mm);
final_resid_var = variance_all(final_resid_ifg_mm);
raw_def_misfit_var = variance_all(raw_def_misfit_mm);

metrics = struct();
metrics.reference_free = true;
metrics.message = 'Reference-free diagnostics were computed.';
metrics.model_fit_rmse_mm = rms_zero(final_resid_ifg_mm);
metrics.model_fit_mae_mm = mean_abs_zero(final_resid_ifg_mm);
metrics.model_fit_p90_abs_mm = percentile_abs(final_resid_ifg_mm, 90);
metrics.branch_model_fit_rmse_mm = struct( ...
    'filter', rms_zero(filter_resid_ifg_mm), ...
    'corrected', rms_zero(corrected_resid_ifg_mm));
metrics.branch_model_fit_mae_mm = struct( ...
    'filter', mean_abs_zero(filter_resid_ifg_mm), ...
    'corrected', mean_abs_zero(corrected_resid_ifg_mm));
metrics.variance_explained = max(0.0, 1.0 - final_resid_var / max(obs_var, 1e-9));
metrics.turbulence_correction_gain = max(0.0, 1.0 - final_resid_var / max(raw_def_misfit_var, 1e-9));
metrics.mean_observation_std_mm = mean(result.uncertainty.observation_std_mm, 'omitnan');
metrics.mean_model_uncertainty = mean(result.uncertainty.model_uncertainty, 'omitnan');
metrics.mean_turbulence_ifg_std_mm = sqrt(max(mean(result.kriging.phi_turb_var(:), 'omitnan'), 0)) * abs(data.scale_rad2mm);
metrics.mean_turbulence_seq_std_mm = mean(std(result.phi_turb_seq_mm, 0, 1, 'omitnan'), 'omitnan');
metrics.active_pixel_index = find_active_pixel_reference_free( ...
    result.phi_seq_corrected_mm);
metrics.active_pixel_amplitude_mm = percentile( ...
    result.phi_seq_corrected_mm(metrics.active_pixel_index, :), 95) - ...
    percentile(result.phi_seq_corrected_mm( ...
    metrics.active_pixel_index, :), 5);
metrics.final_epoch_deformation_range_mm = percentile( ...
    result.phi_seq_corrected_mm(:, end), 95) - ...
    percentile(result.phi_seq_corrected_mm(:, end), 5);
metrics.branch_final_epoch_deformation_range_mm = struct( ...
    'filter', percentile(result.phi_seq_filter_mm(:, end), 95) - percentile(result.phi_seq_filter_mm(:, end), 5), ...
    'corrected', metrics.final_epoch_deformation_range_mm);
metrics.final_epoch_turbulence_std_mm = std(result.phi_turb_seq_mm(:, end), 'omitnan');
metrics.turbulence_spatial_roughness_mm_per_km = spatial_roughness(data.coord_km, result.phi_turb_seq_mm(:, end));
metrics.deformation_spatial_roughness_mm_per_km = spatial_roughness( ...
    data.coord_km, result.phi_seq_corrected_mm(:, end));
metrics.final_diagnostic_objective = ...
    result.diagnostic_objective_history(end);
if isfield(result, 'convergence_history') && ...
        ~isempty(result.convergence_history.C_diagnostic)
    metrics.final_convergence_C = ...
        result.convergence_history.C_diagnostic(end);
    metrics.final_convergence_Delta_D = result.convergence_history.Delta_D(end);
    metrics.final_convergence_Delta_tau = ...
        result.convergence_history.Delta_tau(end);
    metrics.final_convergence_Delta_E = ...
        result.convergence_history.Delta_E(end);
    metrics.final_convergence_Delta_R = result.convergence_history.Delta_R(end);
end

if isfield(result, 'runtime_profile') && isfield(result.runtime_profile, 'avg_kriging_neighbors')
    metrics.avg_kriging_neighbors = result.runtime_profile.avg_kriging_neighbors;
else
    metrics.avg_kriging_neighbors = NaN;
end

if isfield(result, 'kriging') && isfield(result.kriging, 'quality')
    q = result.kriging.quality;
    metrics.kriging_fallback_count = get_struct_field(q, 'fallback_count', 0);
    metrics.kriging_clipped_count = get_struct_field(q, 'clipped_count', 0);
else
    metrics.kriging_fallback_count = 0;
    metrics.kriging_clipped_count = 0;
end
end

function metrics = add_truth_based_metrics(metrics, data, result)
truth_def = data.truth.phi_def_mm;
truth_turb = data.truth.phi_turb_mm;
est_def = result.phi_seq_corrected_mm;
est_def_filter = result.phi_seq_filter_mm;
est_turb = result.phi_turb_seq_mm;

metrics.available = true;
metrics.deformation_rmse_mm = rmse(est_def, truth_def);
metrics.deformation_rmse_filter_mm = rmse(est_def_filter, truth_def);
metrics.deformation_rmse_corrected_mm = metrics.deformation_rmse_mm;
metrics.deformation_mae_mm = mae(est_def, truth_def);
metrics.deformation_mae_filter_mm = mae(est_def_filter, truth_def);
metrics.deformation_mae_corrected_mm = metrics.deformation_mae_mm;
metrics.deformation_final_epoch_rmse_mm = rmse(est_def(:, end), truth_def(:, end));
metrics.deformation_final_epoch_mae_mm = mae(est_def(:, end), truth_def(:, end));

metrics.turbulence_rmse_mm = rmse(est_turb, truth_turb);
metrics.turbulence_mae_mm = mae(est_turb, truth_turb);
metrics.turbulence_final_epoch_rmse_mm = rmse(est_turb(:, end), truth_turb(:, end));
metrics.turbulence_final_epoch_mae_mm = mae(est_turb(:, end), truth_turb(:, end));

if isfield(data.truth, 'phi_turb_ifg_rad')
    est_ifg_turb_mm = result.phi_turb_final_rad * data.scale_rad2mm;
    truth_ifg_turb_mm = data.truth.phi_turb_ifg_rad * data.scale_rad2mm;
    metrics.turbulence_ifg_rmse_mm = rmse(est_ifg_turb_mm, truth_ifg_turb_mm);
    metrics.turbulence_ifg_mae_mm = mae(est_ifg_turb_mm, truth_ifg_turb_mm);
end

deformation_error = est_def - truth_def;
turbulence_error = est_turb - truth_turb;
metrics.time_series_mae_per_epoch_mm = mean(abs(deformation_error), 1, 'omitnan');
metrics.time_series_rmse_per_epoch_mm = sqrt(mean(deformation_error.^2, 1, 'omitnan'));
metrics.turbulence_mae_per_epoch_mm = mean(abs(turbulence_error), 1, 'omitnan');
metrics.turbulence_rmse_per_epoch_mm = sqrt(mean(turbulence_error.^2, 1, 'omitnan'));

metrics.active_pixel_index = find_active_pixel_reference_free(truth_def);
metrics.active_pixel_mae_mm = mae(est_def(metrics.active_pixel_index, :), truth_def(metrics.active_pixel_index, :));
metrics.active_pixel_rmse_mm = rmse(est_def(metrics.active_pixel_index, :), truth_def(metrics.active_pixel_index, :));

truth_def_ifg_mm = sequence_to_ifg( ...
    truth_def, data.ifg_pairs, 1);
estimated_turb_ifg_mm = result.turbulence_final .* data.scale_rad2mm;
truth_turb_ifg_mm = sequence_to_ifg( ...
    truth_turb, data.ifg_pairs, 1);
turbulence_error_ifg_mm = estimated_turb_ifg_mm - truth_turb_ifg_mm;
metrics.deformation_to_turbulence_leakage_correlation = ...
    vector_correlation(turbulence_error_ifg_mm, truth_def_ifg_mm);
metrics.deformation_to_turbulence_leakage_slope = ...
    vector_slope(turbulence_error_ifg_mm, truth_def_ifg_mm);
metrics.network_corrected_deformation_rmse_mm = rmse( ...
    result.timeseries_atm_corrected, truth_def);
end

function write_metrics_summary(metrics, file_path)
if metrics.has_truth
    lines = truth_summary_lines(metrics);
else
    lines = reference_free_summary_lines(metrics);
end

fid = fopen(file_path, 'w', 'n', 'UTF-8');
if fid < 0
    return;
end
for i = 1:numel(lines)
    fprintf(fid, '%s\n', lines{i});
end
fclose(fid);
end

function lines = truth_summary_lines(metrics)
lines = {
    '# Quantitative Evaluation Summary'
    ''
    sprintf('- Deformation RMSE, filter: %.4f mm', metrics.deformation_rmse_filter_mm)
    sprintf('- Deformation RMSE, STAR-KF joint: %.4f mm', metrics.deformation_rmse_corrected_mm)
    sprintf('- Deformation RMSE, atmosphere-corrected network only: %.4f mm', metrics.network_corrected_deformation_rmse_mm)
    sprintf('- Deformation MAE, filter: %.4f mm', metrics.deformation_mae_filter_mm)
    sprintf('- Deformation MAE, STAR-KF iterback: %.4f mm', metrics.deformation_mae_corrected_mm)
    sprintf('- Corrected deformation final-epoch RMSE: %.4f mm', metrics.deformation_final_epoch_rmse_mm)
    sprintf('- Turbulence SLC RMSE: %.4f mm', metrics.turbulence_rmse_mm)
    sprintf('- Turbulence SLC MAE: %.4f mm', metrics.turbulence_mae_mm)
    sprintf('- Turbulence final-epoch RMSE: %.4f mm', metrics.turbulence_final_epoch_rmse_mm)
    sprintf('- Active pixel RMSE: %.4f mm', metrics.active_pixel_rmse_mm)
    sprintf('- Deformation-to-turbulence leakage correlation: %.4f', metrics.deformation_to_turbulence_leakage_correlation)
    sprintf('- Deformation-to-turbulence leakage slope: %.4f', metrics.deformation_to_turbulence_leakage_slope)
    sprintf('- Mean posterior observation STD: %.4f mm', metrics.mean_observation_std_mm)
    sprintf('- Mean model uncertainty: %.4f', metrics.mean_model_uncertainty)
    ''
    'Reference-free diagnostics:'
    sprintf('- Modeled IFG residual RMSE, filter: %.4f mm', metrics.branch_model_fit_rmse_mm.filter)
    sprintf('- Modeled IFG residual RMSE, corrected: %.4f mm', metrics.branch_model_fit_rmse_mm.corrected)
    sprintf('- Variance explained by deformation+turbulence model: %.2f %%', 100 * metrics.variance_explained)
    sprintf('- Turbulence correction gain against deformation-only fit: %.2f %%', 100 * metrics.turbulence_correction_gain)
    ''
    'Time-series MAE per epoch (mm):'
    mat2str(metrics.time_series_mae_per_epoch_mm, 4)
    ''
    'Turbulence RMSE per epoch (mm):'
    mat2str(metrics.turbulence_rmse_per_epoch_mm, 4)
    };
if isfield(metrics, 'final_convergence_C')
    lines{end + 1} = '';
    lines{end + 1} = 'Convergence diagnostics:';
    lines{end + 1} = sprintf('- Final C: %.4f', metrics.final_convergence_C);
    lines{end + 1} = sprintf('- Final Delta_D: %.4f', metrics.final_convergence_Delta_D);
    lines{end + 1} = sprintf('- Final Delta_tau: %.4f', metrics.final_convergence_Delta_tau);
    lines{end + 1} = sprintf('- Final Delta_E: %.4f', metrics.final_convergence_Delta_E);
    lines{end + 1} = sprintf('- Final Delta_R: %.4f', metrics.final_convergence_Delta_R);
end
end

function lines = reference_free_summary_lines(metrics)
lines = {
    '# Reference-Free Evaluation Summary'
    ''
    'Ground truth is unavailable, so the values below are internal consistency and uncertainty diagnostics rather than absolute accuracy metrics.'
    ''
    sprintf('- Modeled IFG residual RMSE, filter: %.4f mm', metrics.branch_model_fit_rmse_mm.filter)
    sprintf('- Modeled IFG residual RMSE, corrected: %.4f mm', metrics.branch_model_fit_rmse_mm.corrected)
    sprintf('- Modeled IFG residual MAE, filter: %.4f mm', metrics.branch_model_fit_mae_mm.filter)
    sprintf('- Modeled IFG residual MAE, corrected: %.4f mm', metrics.branch_model_fit_mae_mm.corrected)
    sprintf('- 90th percentile absolute IFG residual: %.4f mm', metrics.model_fit_p90_abs_mm)
    sprintf('- Variance explained by deformation+turbulence model: %.2f %%', 100 * metrics.variance_explained)
    sprintf('- Turbulence correction gain against deformation-only fit: %.2f %%', 100 * metrics.turbulence_correction_gain)
    sprintf('- Mean posterior observation STD: %.4f mm', metrics.mean_observation_std_mm)
    sprintf('- Mean model uncertainty: %.4f', metrics.mean_model_uncertainty)
    sprintf('- Mean kriging IFG STD: %.4f mm', metrics.mean_turbulence_ifg_std_mm)
    sprintf('- Mean SLC turbulence epoch STD: %.4f mm', metrics.mean_turbulence_seq_std_mm)
    sprintf('- Final-epoch deformation 5-95%% range, filter: %.4f mm', metrics.branch_final_epoch_deformation_range_mm.filter)
    sprintf('- Final-epoch deformation 5-95%% range, corrected: %.4f mm', metrics.branch_final_epoch_deformation_range_mm.corrected)
    sprintf('- Final-epoch turbulence STD: %.4f mm', metrics.final_epoch_turbulence_std_mm)
    sprintf('- Final-epoch turbulence spatial roughness: %.4f mm/km', metrics.turbulence_spatial_roughness_mm_per_km)
    sprintf('- Final-epoch deformation spatial roughness: %.4f mm/km', metrics.deformation_spatial_roughness_mm_per_km)
    sprintf('- Active-pixel deformation amplitude (5-95%%): %.4f mm', metrics.active_pixel_amplitude_mm)
    sprintf('- Average kriging neighbors: %.2f', metrics.avg_kriging_neighbors)
    sprintf('- Kriging fallback count: %d', metrics.kriging_fallback_count)
    sprintf('- Kriging clipped count: %d', metrics.kriging_clipped_count)
    sprintf('- Final diagnostic objective (not a joint optimization objective): %.4f', metrics.final_diagnostic_objective)
    };
if isfield(metrics, 'final_convergence_C')
    lines{end + 1} = sprintf('- Final convergence C: %.4f', metrics.final_convergence_C);
    lines{end + 1} = sprintf('- Final convergence Delta_D: %.4f', metrics.final_convergence_Delta_D);
    lines{end + 1} = sprintf('- Final convergence Delta_tau: %.4f', metrics.final_convergence_Delta_tau);
    lines{end + 1} = sprintf('- Final convergence Delta_E: %.4f', metrics.final_convergence_Delta_E);
    lines{end + 1} = sprintf('- Final convergence Delta_R: %.4f', metrics.final_convergence_Delta_R);
end
end

function value = rmse(a, b)
diff_val = a - b;
value = sqrt(mean(diff_val(:).^2, 'omitnan'));
end

function value = mae(a, b)
diff_val = a - b;
value = mean(abs(diff_val(:)), 'omitnan');
end

function value = rms_zero(a)
value = sqrt(mean(a(:).^2, 'omitnan'));
end

function value = mean_abs_zero(a)
value = mean(abs(a(:)), 'omitnan');
end

function value = variance_all(a)
valid = a(isfinite(a));
if isempty(valid)
    value = NaN;
else
    value = var(valid, 1);
end
end

function idx = find_active_pixel_reference_free(phi_seq_mm)
activity = var(phi_seq_mm, 0, 2, 'omitnan');
[~, idx] = max(activity);
end

function value = spatial_roughness(coord_km, values)
valid = isfinite(values);
coord_use = coord_km(valid, :);
values = values(valid);
if numel(values) < 8
    value = NaN;
    return;
end

try
    [idx, dist] = knnsearch(coord_use, coord_use, 'K', min(7, size(coord_use, 1)));
    idx = idx(:, 2:end);
    dist = max(dist(:, 2:end), 1e-6);
catch
    D = pdist2(coord_use, coord_use);
    [dist, idx] = sort(D, 2, 'ascend');
    idx = idx(:, 2:min(7, size(coord_use, 1)));
    dist = max(dist(:, 2:min(7, size(coord_use, 1))), 1e-6);
end

neighbor_vals = values(idx);
rough = abs(values - mean(neighbor_vals, 2, 'omitnan')) ./ mean(dist, 2, 'omitnan');
value = mean(rough, 'omitnan');
end

function value = percentile(a, p)
valid = a(isfinite(a));
if isempty(valid)
    value = NaN;
else
    value = prctile(valid, p);
end
end

function value = percentile_abs(a, p)
valid = abs(a(isfinite(a)));
if isempty(valid)
    value = NaN;
else
    value = prctile(valid, p);
end
end

function value = get_struct_field(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end

function value = vector_correlation(a, b)
a = a(:);
b = b(:);
valid = isfinite(a) & isfinite(b);
a = a(valid);
b = b(valid);
if numel(a) < 3
    value = NaN;
    return;
end
a = a - mean(a, 'omitnan');
b = b - mean(b, 'omitnan');
value = sum(a .* b) / max(sqrt(sum(a .^ 2) * sum(b .^ 2)), eps);
end

function value = vector_slope(response, predictor)
response = response(:);
predictor = predictor(:);
valid = isfinite(response) & isfinite(predictor);
response = response(valid);
predictor = predictor(valid);
predictor = predictor - mean(predictor, 'omitnan');
response = response - mean(response, 'omitnan');
value = sum(response .* predictor) / max(sum(predictor .^ 2), eps);
end
