function [D_leak_iter_mm, weights, diagnostics] = ...
    iteration_leakage_recovery( ...
    A_iter_seq_mm, D_filter_mm, D_anchor_mm, z_d_raw_mm, ...
    cfg, acq_times, i_master)
%ITERATION_LEAKAGE_RECOVERY Extract deformation-like turbulence content.

if nargin < 5 || isempty(cfg)
    cfg = struct();
end
if nargin < 6 || isempty(acq_times)
    acq_times = 1:size(A_iter_seq_mm, 2);
end
if nargin < 7 || isempty(i_master)
    i_master = 1;
end

if get_param(cfg, 'use_imm_extraction', true)
    imm_result = extract_leakage_imm_kalman( ...
        A_iter_seq_mm, acq_times, cfg, i_master);
    D_candidate_mm = imm_result.D_candidate_mm;
    extraction_diagnostics = imm_result.diagnostics;
else
    D_candidate_mm = A_iter_seq_mm;
    extraction_diagnostics = struct('method', 'direct_sequence');
end

D_candidate_mm = rebase_to_master(D_candidate_mm, i_master);
[weights, weight_diagnostics] = build_iteration_leakage_weights( ...
    D_candidate_mm, D_filter_mm, D_anchor_mm, z_d_raw_mm, cfg, ...
    i_master);

beta_min = max(get_param(cfg, 'beta_min', 1.0), 0);
beta_max = max(get_param(cfg, 'beta_max', 1.25), beta_min);
beta = beta_min + (beta_max - beta_min) .* weights.W_total;

D_candidate_mm(~isfinite(D_candidate_mm)) = 0;
D_leak_iter_mm = beta .* D_candidate_mm;
D_leak_iter_mm = rebase_to_master(D_leak_iter_mm, i_master);

diagnostics = struct( ...
    'extraction', extraction_diagnostics, ...
    'weights', weight_diagnostics, ...
    'candidate_rms_mm', rms_all(D_candidate_mm), ...
    'leak_rms_mm', rms_all(D_leak_iter_mm), ...
    'beta_mean', mean(beta, 'omitnan'), ...
    'beta_p95', finite_percentile(beta, 95));
end

function values = rebase_to_master(values, i_master)
i_master = min(max(round(i_master), 1), size(values, 2));
values = values - values(:, i_master);
end

function value = rms_all(values)
value = sqrt(mean(values(:) .^ 2, 'omitnan'));
if ~isfinite(value), value = 0; end
end

function value = finite_percentile(values, p)
valid = values(isfinite(values));
if isempty(valid)
    value = NaN;
else
    value = prctile(valid, p);
end
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
