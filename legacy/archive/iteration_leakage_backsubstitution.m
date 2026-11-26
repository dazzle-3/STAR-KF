function [kriging, diagnostics] = iteration_leakage_backsubstitution( ...
    kriging, filter_result, D_anchor_mm, data, cfg, iteration, ...
    updated_r, covariance_components)
%ITERATION_LEAKAGE_BACKSUBSTITUTION Remove guarded leakage from turbulence.

diagnostics = default_diagnostics();
iter_cfg = get_section(cfg, 'iter_leakrec');
if ~get_param(iter_cfg, 'enable', false) || ...
        iteration < get_param(iter_cfg, 'start_iteration', 1)
    diagnostics.stop_reason = 'disabled';
    return;
end

diagnostics.enabled = true;
i_master = min(max(round(cfg.i_master), 1), data.n_slc);
opt = get_section(cfg, 'optimization');
max_inner_iters = max(round(get_param(iter_cfg, 'max_inner_iters', 2)), 1);
D_add_total_mm = zeros(size(filter_result.phi_seq_mm));
last_weights = [];

for inner = 1:max_inner_iters
    A_iter_seq_mm = recover_sequence_with_random_model( ...
        kriging.phi_turb, data, i_master, updated_r, ...
        covariance_components, opt) .* data.scale_rad2mm;
    [D_leak_iter_mm, weights, leak_diag] = ...
        iteration_leakage_recovery( ...
        A_iter_seq_mm, filter_result.phi_seq_mm, D_anchor_mm, ...
        data.z_d_raw, iter_cfg, data.acq_times, i_master);

    active = weights.W_total >= ...
        get_param(iter_cfg, 'min_total_weight_active', 0.35);
    active = active & all(isfinite(D_leak_iter_mm), 2);
    active_fraction = mean(active);
    support_corr = weights.support_corr;
    median_support_corr = finite_median(support_corr(active));

    diagnostics.active_fraction = active_fraction;
    diagnostics.median_support_corr = median_support_corr;
    last_weights = weights;

    if active_fraction < ...
            get_param(iter_cfg, 'stop_active_fraction', 0.03)
        diagnostics.stop_reason = 'active_fraction';
        break;
    end
    if ~isfinite(median_support_corr) || median_support_corr < ...
            get_param(iter_cfg, 'stop_support_corr', 0.15)
        diagnostics.stop_reason = 'support_correlation';
        break;
    end

    alpha = get_param(iter_cfg, 'alpha_max', 0.35) .* ...
        weights.W_total;
    alpha(~active) = 0;
    alpha(active) = max(alpha(active), ...
        get_param(iter_cfg, 'alpha_min', 0.10));
    alpha = min(max(alpha, 0), ...
        get_param(iter_cfg, 'alpha_max', 0.35));
    D_add_iter_mm = alpha .* D_leak_iter_mm;
    D_add_iter_mm = rebase_to_master(D_add_iter_mm, i_master);

    eta_iter = rms_subset(D_add_iter_mm, active) / ...
        max(rms_subset(filter_result.phi_seq_mm, active), eps);
    endpoint_add = abs(D_add_iter_mm(:, end) - ...
        D_add_iter_mm(:, i_master));
    median_endpoint_add = finite_median(endpoint_add(active));
    diagnostics.eta_iter = eta_iter;
    diagnostics.median_endpoint_add = median_endpoint_add;
    diagnostics.alpha_mean = finite_mean(alpha(active));
    diagnostics.alpha_p95 = finite_percentile(alpha(active), 95);

    phi_leak_ifg_rad = sequence_to_ifg( ...
        D_add_iter_mm, data.ifg_pairs, 1 / data.scale_rad2mm);
    phi_turb_before = kriging.phi_turb;
    phi_turb_trial = phi_turb_before - phi_leak_ifg_rad;
    residual_worsening = compute_residual_worsening( ...
        phi_turb_before, phi_turb_trial, data, i_master, ...
        updated_r, covariance_components, opt);

    max_worsening = get_param( ...
        iter_cfg, 'max_residual_worsening', 0.015);
    if residual_worsening > max_worsening
        alpha = 0.5 .* alpha;
        D_add_iter_mm = alpha .* D_leak_iter_mm;
        D_add_iter_mm = rebase_to_master(D_add_iter_mm, i_master);
        phi_leak_ifg_rad = sequence_to_ifg( ...
            D_add_iter_mm, data.ifg_pairs, 1 / data.scale_rad2mm);
        phi_turb_trial = phi_turb_before - phi_leak_ifg_rad;
        residual_worsening = compute_residual_worsening( ...
            phi_turb_before, phi_turb_trial, data, i_master, ...
            updated_r, covariance_components, opt);
        eta_iter = rms_subset(D_add_iter_mm, active) / ...
            max(rms_subset(filter_result.phi_seq_mm, active), eps);
        endpoint_add = abs(D_add_iter_mm(:, end) - ...
            D_add_iter_mm(:, i_master));
        median_endpoint_add = finite_median(endpoint_add(active));
        diagnostics.eta_iter = eta_iter;
        diagnostics.median_endpoint_add = median_endpoint_add;
        diagnostics.alpha_mean = finite_mean(alpha(active));
        diagnostics.alpha_p95 = finite_percentile(alpha(active), 95);
    end

    diagnostics.residual_worsening = residual_worsening;
    if residual_worsening > max_worsening
        diagnostics.stop_reason = 'residual_worsening_rejected';
        break;
    end

    kriging.phi_turb = phi_turb_trial;
    D_add_total_mm = D_add_total_mm + D_add_iter_mm;
    diagnostics.num_inner_iters = diagnostics.num_inner_iters + 1;
    diagnostics.D_add_rms = rms_subset(D_add_total_mm, true(size(active)));
    diagnostics.leak_candidate_rms = leak_diag.leak_rms_mm;
    diagnostics.stop_reason = 'max_inner_iters';

    if eta_iter < get_param(iter_cfg, 'stop_rel_increment', 0.02)
        diagnostics.stop_reason = 'relative_increment';
        break;
    end
    if median_endpoint_add < ...
            get_param(iter_cfg, 'stop_endpoint_mm', 1.0)
        diagnostics.stop_reason = 'endpoint_increment';
        break;
    end
end

if get_param(iter_cfg, 'save_spatial_diagnostics', false) && ...
        ~isempty(last_weights)
    diagnostics.endpoint_add = abs(D_add_total_mm(:, end) - ...
        D_add_total_mm(:, i_master));
    diagnostics.W_total = last_weights.W_total;
    diagnostics.support_corr = last_weights.support_corr;
end
kriging.iterback = diagnostics;
end

function worsening = compute_residual_worsening( ...
    phi_turb_before, phi_turb_after, data, i_master, ...
    updated_r, covariance_components, opt)
before_rms = clean_network_residual_rms( ...
    phi_turb_before, data, i_master, updated_r, ...
    covariance_components, opt);
after_rms = clean_network_residual_rms( ...
    phi_turb_after, data, i_master, updated_r, ...
    covariance_components, opt);
worsening = (after_rms - before_rms) / max(before_rms, eps);
if ~isfinite(worsening)
    worsening = inf;
end
end

function value = clean_network_residual_rms( ...
    phi_turb, data, i_master, updated_r, covariance_components, opt)
phi_clean = data.phi_mat - phi_turb;
D_clean_rad = recover_sequence_with_random_model( ...
    phi_clean, data, i_master, updated_r, covariance_components, opt);
phi_reconstructed = sequence_to_ifg( ...
    D_clean_rad, data.ifg_pairs, 1);
residual = phi_clean - phi_reconstructed;
value = sqrt(mean(residual(:) .^ 2, 'omitnan'));
end

function values = rebase_to_master(values, i_master)
values = values - values(:, i_master);
end

function value = rms_subset(values, rows)
if ~any(rows)
    value = 0;
    return;
end
subset = values(rows, :);
value = sqrt(mean(subset(:) .^ 2, 'omitnan'));
if ~isfinite(value), value = 0; end
end

function value = finite_mean(values)
value = mean(values, 'omitnan');
if ~isfinite(value), value = 0; end
end

function value = finite_median(values)
value = median(values, 'omitnan');
if ~isfinite(value), value = NaN; end
end

function value = finite_percentile(values, p)
valid = values(isfinite(values));
if isempty(valid)
    value = NaN;
else
    value = prctile(valid, p);
end
end

function section = get_section(cfg, name)
if isstruct(cfg) && isfield(cfg, name) && isstruct(cfg.(name))
    section = cfg.(name);
else
    section = struct();
end
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end

function diagnostics = default_diagnostics()
diagnostics = struct( ...
    'enabled', false, ...
    'num_inner_iters', 0, ...
    'active_fraction', 0, ...
    'eta_iter', NaN, ...
    'median_endpoint_add', NaN, ...
    'median_support_corr', NaN, ...
    'residual_worsening', NaN, ...
    'alpha_mean', 0, ...
    'alpha_p95', 0, ...
    'D_add_rms', 0, ...
    'leak_candidate_rms', 0, ...
    'endpoint_add', [], ...
    'W_total', [], ...
    'support_corr', [], ...
    'stop_reason', 'not_started');
end
