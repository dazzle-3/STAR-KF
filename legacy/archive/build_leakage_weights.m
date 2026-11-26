function [weights, diagnostics] = build_leakage_weights( ...
    D_filter_mm, D_candidate_mm, cfg, imm_result, D_anchor_mm)
if nargin < 3 || isempty(cfg)
    cfg = struct();
end
if nargin < 4
    imm_result = struct();
end
if nargin < 5
    D_anchor_mm = [];
end

validate_same_size(D_filter_mm, D_candidate_mm);

W_def = build_deformation_gate(D_filter_mm, cfg);
W_sign = build_direction_weight(D_filter_mm, D_candidate_mm, cfg);
[W_dyn, W_imm, W_smooth] = build_dynamic_weight(D_candidate_mm, cfg, imm_result);
W_total = combine_leakage_weights(W_def, W_sign, W_dyn, cfg);
[W_total, endpoint_gate] = apply_endpoint_direction_gate( ...
    W_total, D_filter_mm, D_candidate_mm, D_anchor_mm, cfg);

weights = struct( ...
    'W_def', W_def, ...
    'W_sign', W_sign, ...
    'W_dyn', W_dyn, ...
    'W_total', W_total, ...
    'W_imm', W_imm, ...
    'W_smooth', W_smooth);

diagnostics = struct( ...
    'W_def_mean', mean(W_def, 'omitnan'), ...
    'W_sign_mean', mean(W_sign, 'omitnan'), ...
    'W_dyn_mean', mean(W_dyn, 'omitnan'), ...
    'W_total_mean', mean(W_total, 'omitnan'), ...
    'W_imm_mean', mean(W_imm, 'omitnan'), ...
    'W_smooth_mean', mean(W_smooth, 'omitnan'), ...
    'endpoint_gate_pass_fraction', mean(endpoint_gate, 'omitnan'), ...
    'W_total_median', median(W_total, 'omitnan'), ...
    'W_total_p90', prctile(W_total(isfinite(W_total)), 90));
end

function [W_total, gate] = apply_endpoint_direction_gate( ...
    W_total, D_filter_mm, D_candidate_mm, D_anchor_mm, cfg)
n_pts = size(D_filter_mm, 1);
gate = true(n_pts, 1);
candidate_endpoint = D_candidate_mm(:, end) - D_candidate_mm(:, 1);

if get_param(cfg, 'endpoint_sign_gate', false)
    filter_endpoint = D_filter_mm(:, end) - D_filter_mm(:, 1);
    gate = gate & filter_endpoint .* candidate_endpoint > 0;
end

if get_param(cfg, 'anchor_sign_gate', false) && ...
        ~isempty(D_anchor_mm) && isequal(size(D_anchor_mm), size(D_filter_mm))
    anchor_endpoint = D_anchor_mm(:, end) - D_anchor_mm(:, 1);
    gate = gate & anchor_endpoint .* candidate_endpoint > 0;
end

gate(~isfinite(candidate_endpoint)) = false;
W_total(~gate) = 0;
end

function W_total = combine_leakage_weights(W_def, W_sign, W_dyn, cfg)
mode = lower(char(get_param(cfg, 'weight_combine_mode', 'product')));
switch mode
    case {'soft_product', 'soft', 'guarded_product'}
        def_power = max(get_param(cfg, 'def_power', 1.0), 0);
        sign_power = max(get_param(cfg, 'sign_power', 1.0), 0);
        dyn_power = max(get_param(cfg, 'dyn_power', 1.0), 0);
        W_total = (max(W_def, 0) .^ def_power) .* ...
            (max(W_sign, 0) .^ sign_power) .* ...
            (max(W_dyn, 0) .^ dyn_power);

        active_def_threshold = min(max(get_param(cfg, ...
            'active_def_threshold', 0.6), 0), 1);
        active_sign_threshold = min(max(get_param(cfg, ...
            'active_sign_threshold', 0.25), 0), 1);
        min_active_weight = min(max(get_param(cfg, ...
            'min_total_weight_active', 0.0), 0), 1);
        active = W_def >= active_def_threshold & ...
            W_sign >= active_sign_threshold;
        W_total(active) = max(W_total(active), min_active_weight);

        gain = max(get_param(cfg, 'total_gain', 1.0), 0);
        W_total = gain .* W_total;
        max_total = min(max(get_param(cfg, 'max_total_weight', 1.0), 0), 1);
        W_total = min(W_total, max_total);
    otherwise
        W_total = W_def .* W_sign .* W_dyn;
end
W_total(~isfinite(W_total)) = 0;
W_total = min(max(W_total, 0), 1);
end

function W_def = build_deformation_gate(D_filter_mm, cfg)
low_p = get_param(cfg, 'def_gate_low_percentile', 60);
high_p = get_param(cfg, 'def_gate_high_percentile', 90);
low_p = min(max(low_p, 0), 100);
high_p = min(max(high_p, low_p + 1e-6), 100);

S_filter = D_filter_mm(:, end) - D_filter_mm(:, 1);
abs_S = abs(S_filter);
valid = isfinite(abs_S);
W_def = zeros(size(abs_S));
if ~any(valid)
    return;
end

P_low = prctile(abs_S(valid), low_p);
P_high = prctile(abs_S(valid), high_p);
den = P_high - P_low;
if ~isfinite(den) || den <= 0
    return;
end
W_def(valid) = (abs_S(valid) - P_low) ./ den;
W_def = min(max(W_def, 0), 1);
end

function W_sign = build_direction_weight(D_filter_mm, D_candidate_mm, cfg)
eps_corr = get_param(cfg, 'eps_corr', 1e-9);
n_pts = size(D_filter_mm, 1);
W_sign = zeros(n_pts, 1);

for i = 1:n_pts
    a = D_candidate_mm(i, :);
    b = D_filter_mm(i, :);
    valid = isfinite(a) & isfinite(b);
    if nnz(valid) < 3
        continue;
    end
    a = a(valid) - mean(a(valid), 'omitnan');
    b = b(valid) - mean(b(valid), 'omitnan');
    den = sqrt(sum(a.^2) * sum(b.^2));
    if den <= eps_corr
        continue;
    end
    rho = sum(a .* b) / den;
    W_sign(i) = max(0, rho);
end
W_sign(~isfinite(W_sign)) = 0;
W_sign = min(max(W_sign, 0), 1);
end

function [W_dyn, W_imm, W_smooth] = build_dynamic_weight(D_candidate_mm, cfg, imm_result)
n_pts = size(D_candidate_mm, 1);
if isstruct(imm_result) && isfield(imm_result, 'W_imm') && ~isempty(imm_result.W_imm)
    W_imm = imm_result.W_imm(:);
else
    W_imm = ones(n_pts, 1);
end
if numel(W_imm) ~= n_pts
    W_imm = ones(n_pts, 1);
end
W_imm(~isfinite(W_imm)) = 0;
W_imm = min(max(W_imm, 0), 1);

eps_smooth = get_param(cfg, 'eps_smooth_mm', 1e-6);
seq_amp = row_rms(center_rows(D_candidate_mm));
if size(D_candidate_mm, 2) >= 3
    rough = row_rms(diff(D_candidate_mm, 2, 2));
else
    rough = zeros(n_pts, 1);
end
ratio = rough ./ max(seq_amp, eps_smooth);
W_smooth = 1 - ratio;
W_smooth(~isfinite(W_smooth)) = 0;
W_smooth = min(max(W_smooth, 0), 1);

W_dyn = W_imm .* W_smooth;
W_dyn(~isfinite(W_dyn)) = 0;
W_dyn = min(max(W_dyn, 0), 1);
end

function centered = center_rows(values)
centered = values;
row_mean = mean(values, 2, 'omitnan');
for i = 1:size(values, 1)
    valid = isfinite(values(i, :));
    centered(i, valid) = values(i, valid) - row_mean(i);
    centered(i, ~valid) = NaN;
end
end

function value = row_rms(values)
value = sqrt(mean(values.^2, 2, 'omitnan'));
value(~isfinite(value)) = 0;
end

function validate_same_size(a, b)
if ~isequal(size(a), size(b))
    error('D_filter_mm and D_candidate_mm must have the same size.');
end
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
