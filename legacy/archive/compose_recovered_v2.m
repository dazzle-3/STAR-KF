function result = compose_recovered_v2( ...
    D_base_mm, D_filter_mm, raw_obs_mm, D_leak_mm, cfg)
if nargin < 5 || isempty(cfg)
    cfg = struct();
end
validate_same_size(D_base_mm, D_filter_mm, raw_obs_mm, D_leak_mm);

if ~get_param(cfg, 'output_adaptive_gain_enable', true)
    fixed_gain = max(get_param(cfg, 'output_fixed_gain', 1.25), 0);
    gain = fixed_gain .* ones(size(D_base_mm, 1), 1);
    support_corr = ones(size(gain));
    support_weight = ones(size(gain));
    smoothness = leakage_smoothness(D_leak_mm, cfg);
else
    support_residual = raw_obs_mm - D_base_mm;
    support_corr = row_correlation(D_leak_mm, support_residual);
    corr_low = get_param(cfg, 'output_support_corr_low', 0.25);
    corr_high = get_param(cfg, 'output_support_corr_high', 0.35);
    support_weight = smoothstep_weight( ...
        support_corr, corr_low, corr_high);

    smoothness = leakage_smoothness(D_leak_mm, cfg);
    gain_min = max(get_param(cfg, 'output_gain_min', 1.25), 0);
    gain_max = max(get_param(cfg, 'output_gain_max', 4.55), gain_min);
    smooth_power = max(get_param(cfg, ...
        'output_smoothness_power', 2.0), 0);
    gain = support_weight .* (gain_min + ...
        (gain_max - gain_min) .* smoothness .^ smooth_power);
end

gain = apply_endpoint_cap(gain, D_leak_mm, D_base_mm, ...
    D_filter_mm, raw_obs_mm, cfg);
D_added_mm = bsxfun(@times, D_leak_mm, gain);
D_final_mm = D_base_mm + D_added_mm;

result = struct( ...
    'D_final_mm', D_final_mm, ...
    'D_added_mm', D_added_mm, ...
    'gain', gain, ...
    'support_corr', support_corr, ...
    'support_weight', support_weight, ...
    'smoothness', smoothness, ...
    'diagnostics', struct( ...
        'gain_mean', mean(gain, 'omitnan'), ...
        'gain_p90', prctile(gain(isfinite(gain)), 90), ...
        'active_gain_fraction', mean(gain > 0.05, 'omitnan'), ...
        'support_corr_mean', mean(support_corr, 'omitnan'), ...
        'support_pass_fraction', mean(support_weight > 0.05, 'omitnan')));
end

function gain = apply_endpoint_cap(gain, D_leak, D_base, ...
    D_filter, raw_obs, cfg)
max_fraction = get_param(cfg, ...
    'output_max_added_fraction_of_reference_endpoint', 0.50);
if ~isfinite(max_fraction) || max_fraction <= 0
    return;
end
min_reference = max(get_param(cfg, ...
    'output_min_reference_endpoint_mm', 5.0), 0);
leak_endpoint = D_leak(:, end) - D_leak(:, 1);
base_endpoint = D_base(:, end) - D_base(:, 1);
filter_endpoint = D_filter(:, end) - D_filter(:, 1);
raw_endpoint = raw_obs(:, end) - raw_obs(:, 1);
reference = max([abs(base_endpoint), abs(filter_endpoint), ...
    abs(raw_endpoint), min_reference .* ones(size(base_endpoint))], ...
    [], 2);
added_endpoint = gain .* leak_endpoint;
scale = min(1, max_fraction .* reference ./ ...
    max(abs(added_endpoint), 1e-6));
scale(~isfinite(scale)) = 1;
gain = gain .* scale;
end

function smoothness = leakage_smoothness(values, cfg)
eps_smooth = max(get_param(cfg, 'eps_smooth_mm', 1e-6), 1e-12);
centered = values - mean(values, 2, 'omitnan');
amplitude = sqrt(mean(centered .^ 2, 2, 'omitnan'));
if size(values, 2) >= 3
    roughness = sqrt(mean(diff(values, 2, 2) .^ 2, 2, 'omitnan'));
else
    roughness = zeros(size(amplitude));
end
smoothness = 1 - roughness ./ max(amplitude, eps_smooth);
smoothness(~isfinite(smoothness)) = 0;
smoothness = min(max(smoothness, 0), 1);
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
rho(~isfinite(rho)) = 0;
rho = min(max(rho, -1), 1);
end

function weight = smoothstep_weight(value, low, high)
if ~isfinite(low)
    low = 0.25;
end
if ~isfinite(high) || high <= low
    high = low + 0.10;
end
x = min(max((value - low) ./ (high - low), 0), 1);
weight = x .^ 2 .* (3 - 2 .* x);
end

function validate_same_size(varargin)
base_size = size(varargin{1});
for i = 2:nargin
    if ~isequal(size(varargin{i}), base_size)
        error('All recovered-v2 input sequences must have the same size.');
    end
end
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
