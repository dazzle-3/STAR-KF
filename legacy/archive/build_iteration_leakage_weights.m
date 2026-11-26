function [weights, diagnostics] = build_iteration_leakage_weights( ...
    D_candidate_mm, D_filter_mm, D_anchor_mm, z_d_raw_mm, cfg, i_master)
%BUILD_ITERATION_LEAKAGE_WEIGHTS Conservative per-pixel iterback weights.

if nargin < 5 || isempty(cfg)
    cfg = struct();
end
if nargin < 6 || isempty(i_master)
    i_master = 1;
end
validate_sequences(D_candidate_mm, D_filter_mm, D_anchor_mm, z_d_raw_mm);
i_master = min(max(round(i_master), 1), size(D_candidate_mm, 2));

candidate_endpoint = endpoint(D_candidate_mm, i_master);
filter_endpoint = endpoint(D_filter_mm, i_master);
if isempty(D_anchor_mm)
    anchor_endpoint = filter_endpoint;
else
    anchor_endpoint = endpoint(D_anchor_mm, i_master);
end

gate_signal = max(abs(candidate_endpoint), abs(anchor_endpoint));
W_def = percentile_gate(gate_signal, ...
    get_param(cfg, 'def_gate_low_percentile', 50), ...
    get_param(cfg, 'def_gate_high_percentile', 85));

W_sign = double(candidate_endpoint .* filter_endpoint > 0);
if ~get_param(cfg, 'endpoint_sign_gate', true)
    W_sign(:) = 1;
end

W_anchor = double(candidate_endpoint .* anchor_endpoint > 0);
if isempty(D_anchor_mm) || ~get_param(cfg, 'anchor_sign_gate', true)
    W_anchor(:) = 1;
end

support_signal = z_d_raw_mm - D_filter_mm;
support_corr = row_correlation(D_candidate_mm, support_signal);
W_support = linear_gate(support_corr, ...
    get_param(cfg, 'support_corr_low', 0.15), ...
    get_param(cfg, 'support_corr_high', 0.30));

candidate_centered = D_candidate_mm - ...
    mean(D_candidate_mm, 2, 'omitnan');
candidate_scale = row_rms(candidate_centered);
if size(D_candidate_mm, 2) >= 3
    roughness = row_rms(diff(D_candidate_mm, 2, 2));
else
    roughness = zeros(size(candidate_scale));
end
roughness_ratio = roughness ./ max(candidate_scale, 1e-6);
W_smooth = 1 ./ (1 + roughness_ratio .^ 2);
W_smooth(~isfinite(W_smooth)) = 0;

W_total = W_def .* W_sign .* W_anchor .* W_support .* W_smooth;
W_total = get_param(cfg, 'total_gain', 1.0) .* W_total;
W_total(~isfinite(W_total)) = 0;
W_total = min(max(W_total, 0), 1);

weights = struct( ...
    'W_def', W_def, ...
    'W_sign', W_sign, ...
    'W_anchor', W_anchor, ...
    'W_support', W_support, ...
    'W_smooth', W_smooth, ...
    'W_total', W_total, ...
    'support_corr', support_corr);

diagnostics = struct( ...
    'W_def_mean', finite_mean(W_def), ...
    'W_sign_mean', finite_mean(W_sign), ...
    'W_anchor_mean', finite_mean(W_anchor), ...
    'W_support_mean', finite_mean(W_support), ...
    'W_smooth_mean', finite_mean(W_smooth), ...
    'W_total_mean', finite_mean(W_total), ...
    'W_total_p95', finite_percentile(W_total, 95), ...
    'median_support_corr', finite_median(support_corr));
end

function values = endpoint(sequence, i_master)
values = sequence(:, end) - sequence(:, i_master);
end

function gate = percentile_gate(values, low_percentile, high_percentile)
gate = zeros(size(values));
valid = isfinite(values);
if ~any(valid)
    return;
end
low_percentile = min(max(low_percentile, 0), 100);
high_percentile = min(max(high_percentile, low_percentile + 1e-6), 100);
low_value = prctile(values(valid), low_percentile);
high_value = prctile(values(valid), high_percentile);
if high_value > low_value
    gate(valid) = (values(valid) - low_value) ./ (high_value - low_value);
else
    gate(valid & values > 0) = 1;
end
gate = min(max(gate, 0), 1);
end

function gate = linear_gate(values, low_value, high_value)
if high_value <= low_value
    high_value = low_value + 1e-6;
end
gate = (values - low_value) ./ (high_value - low_value);
gate(~isfinite(gate)) = 0;
gate = min(max(gate, 0), 1);
end

function rho = row_correlation(a, b)
a = a - mean(a, 2, 'omitnan');
b = b - mean(b, 2, 'omitnan');
valid = isfinite(a) & isfinite(b);
a(~valid) = 0;
b(~valid) = 0;
numerator = sum(a .* b, 2);
denominator = sqrt(sum(a .^ 2, 2) .* sum(b .^ 2, 2));
rho = numerator ./ max(denominator, 1e-12);
rho(sum(valid, 2) < 3 | denominator <= 1e-12) = NaN;
rho = min(max(rho, -1), 1);
end

function value = row_rms(values)
value = sqrt(mean(values .^ 2, 2, 'omitnan'));
value(~isfinite(value)) = 0;
end

function validate_sequences(candidate, filter_seq, anchor, raw)
if ~isequal(size(candidate), size(filter_seq)) || ...
        ~isequal(size(candidate), size(raw))
    error(['Candidate, filter, and raw deformation sequences must ', ...
        'have identical dimensions.']);
end
if ~isempty(anchor) && ~isequal(size(candidate), size(anchor))
    error('D_anchor_mm must match the candidate sequence dimensions.');
end
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

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
