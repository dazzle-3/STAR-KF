function [out, shared_cache] = stomac_covariance( ...
    coord_km, resid_ifg_rad, params, shared_cache, source_mask, known_noise)
if nargin < 6 || isempty(known_noise), known_noise=zeros(1,size(resid_ifg_rad,2)); end
known_noise=known_noise(:)';
if isscalar(known_noise), known_noise=repmat(known_noise,1,size(resid_ifg_rad,2)); end
if numel(known_noise)~=size(resid_ifg_rad,2) || any(~isfinite(known_noise)|known_noise<0)
    error('STAR_KF:KnownNoise','Invalid known IFG noise variance.');
end
if nargin < 4
    shared_cache = [];
end
if nargin < 5 || isempty(source_mask)
    source_mask = true(size(resid_ifg_rad, 1), 1);
else
    source_mask = logical(source_mask(:));
end

t_start = tic;
[n_pts, n_ifg] = size(resid_ifg_rad);
if numel(source_mask) ~= n_pts
    error('source_mask length (%d) does not match point count (%d).', ...
        numel(source_mask), n_pts);
end
c_short = nan(1, n_ifg);
c_long = nan(1, n_ifg);
sigma2 = nan(1, n_ifg);
range_x = nan(1, n_ifg);
range_y = nan(1, n_ifg);
range_long = nan(1, n_ifg);
weather_regime = -ones(1, n_ifg);
anisotropy_ratio = nan(1, n_ifg);

use_shared = get_param(params, 'shared_sample_enable', true);
if use_shared
    target_size = min(get_param(params, 'shared_sample_size', get_param(params, 'n_sample_cov', n_pts)), n_pts);
    shared_cache = prepare_shared_sample(coord_km, shared_cache, target_size, params.max_dist_km);
end
parallel_used = should_use_parallel_covariance(params, n_ifg);
log_progress('COV', ...
    ['Starting covariance estimation: IFGs=%d, shared_sample=%d, ', ...
     'compact_pairs=%d, parallel=%d'], ...
    n_ifg, use_shared, get_cache_pair_count(shared_cache), parallel_used);

if parallel_used
    parfor k = 1:n_ifg
        [c_short(k), c_long(k), sigma2(k), range_x(k), ...
            range_y(k), range_long(k), weather_regime(k), ...
            anisotropy_ratio(k)] = estimate_ifg_covariance( ...
            coord_km, resid_ifg_rad(:, k), params, shared_cache, ...
            source_mask, use_shared, known_noise(k));
    end
else
    for k = 1:n_ifg
        if n_ifg > 1 && is_progress_checkpoint(k, n_ifg, 5)
            log_progress('COV', 'Processing IFG %d/%d (%.0f%%)', ...
                k, n_ifg, 100 * k / n_ifg);
        end
        [c_short(k), c_long(k), sigma2(k), range_x(k), ...
            range_y(k), range_long(k), weather_regime(k), ...
            anisotropy_ratio(k)] = estimate_ifg_covariance( ...
            coord_km, resid_ifg_rad(:, k), params, shared_cache, ...
            source_mask, use_shared, known_noise(k));
    end
end

valid=isfinite(c_short)&isfinite(c_long)&isfinite(sigma2)&isfinite(range_x)&isfinite(range_y)&isfinite(range_long);
if ~all(valid)
    error('STAR_KF:SpatialMomentSupport','Insufficient fixed-sample moment support for IFGs %s.',mat2str(find(~valid)));
end

if is_stationary_model(params)
    local_scale_field = ones(n_pts, 1);
else
    local_scale_field = local_scale( ...
        coord_km, resid_ifg_rad, params.nonstationary_strength);
end
elapsed_sec = toc(t_start);
quality = struct( ...
    'model_variant', normalize_model_variant(params), ...
    'mean_sigma2', mean(sigma2, 'omitnan'), ...
    'shared_sample_used', use_shared, ...
    'shared_sample_size', get_cache_sample_size(shared_cache), ...
    'compact_pair_count', get_cache_pair_count(shared_cache), ...
    'parallel_used', parallel_used, ...
    'elapsed_sec', elapsed_sec);

out = struct('c_short_ifg', c_short, 'c_long_ifg', c_long, 'range_x_ifg', range_x, ...
    'range_y_ifg', range_y, 'range_long_ifg', range_long, 'sigma2_ifg', sigma2, ...
    'known_noise_ifg_rad2', known_noise, 'local_scale_field', local_scale_field, 'weather_regime', weather_regime, ...
    'anisotropy_ratio', anisotropy_ratio, 'quality', quality);
log_progress('COV', 'Covariance estimation finished in %.2f s (mean_sigma2=%.4f)', elapsed_sec, quality.mean_sigma2);
end

function [c_short, c_long, sigma2, range_x, range_y, range_long, ...
    weather_regime, anisotropy_ratio] = estimate_ifg_covariance( ...
    coord_km, r, params, shared_cache, source_mask, use_shared, vI)
c_short = NaN;
c_long = NaN;
sigma2 = NaN;
range_x = NaN;
range_y = NaN;
range_long = NaN;
weather_regime = -1;
anisotropy_ratio = NaN;

valid = isfinite(r) & source_mask;
if sum(valid) < params.min_valid_pts
    return;
end
if use_shared
    [d_vec, dx_vec, dy_vec, cv_vec, rs] = ...
        shared_statistics_from_cache(r, shared_cache, source_mask);
else
    [d_vec, dx_vec, dy_vec, cv_vec, rs] = ...
        local_statistics_from_valid(coord_km(valid, :), r(valid), params);
end
if isempty(d_vec) || numel(d_vec) < 80 || isempty(rs)
    return;
end

model_variant = normalize_model_variant(params);
if strcmp(model_variant, 'single_exponential_isotropic_stationary')
    [c_short, range_x] = fit_single_exponential( ...
        d_vec, cv_vec, rs, params);
    c_long = 1e-8;
    sigma2 = max(var(rs) - c_short, 1e-4);
    range_y = range_x;
    range_long = range_x;
    anisotropy_ratio = 1.0;
    ratio = std(rs) / max(median(abs(rs)), 1e-3);
    weather_regime = 0 + (ratio >= 1.8) + (ratio >= 2.5);
    return;
end

abs_cv = abs(cv_vec);
pos_cv = cv_vec(cv_vec > 0);
if isempty(pos_cv)
    c_short = max(var(rs) * 0.35, 1e-6);
else
    c_short = max(prctile(pos_cv, 70), 1e-6);
end
total_var = max(var(rs), c_short + 1e-6);
if strcmp(model_variant,'single_matern_anisotropic_stationary')
    % Re-estimate under the short + unstructured model on current residuals.
    % Preserve the full model's short-amplitude and directional-range estimators.
    c_long = 0;
    sigma2 = max(total_var - c_short - vI, 1e-4);
else
    c_long = max(total_var * (1 - params.multiscale_mix), 1e-6);
    sigma2 = max(total_var - c_short - c_long - vI, 1e-4);
end

w = abs_cv ./ max(d_vec, params.bin_size_km);
wsum = sum(w);
if ~isfinite(wsum) || wsum<=0, return; end
range_x = min(max(sum(dx_vec .* w) / wsum, ...
    params.min_range_km), params.max_range_km);
range_y = min(max(sum(dy_vec .* w) / wsum, ...
    params.min_range_km), params.max_range_km);
abs_sum = sum(abs_cv);
if ~isfinite(abs_sum) || abs_sum<=0, return; end
range_long = min(max(sum(d_vec .* abs_cv) / abs_sum * ...
    params.long_scale_factor, params.min_range_km), params.max_range_km);
anisotropy_ratio = min(max(min(range_x, range_y) / ...
    max(max(range_x, range_y), 1e-6), params.anisotropy_floor), 1.0);
if strcmp(model_variant, 'dual_matern_isotropic_stationary')
    isotropic_range = sqrt(max(range_x * range_y, 1e-12));
    range_x = isotropic_range;
    range_y = isotropic_range;
    anisotropy_ratio = 1.0;
end
ratio = std(rs) / max(median(abs(rs)), 1e-3);
weather_regime = 0 + (ratio >= 1.8) + (ratio >= 2.5);
end

function [C0, d0] = fit_single_exponential( ...
        distance_km, covariance_values, residual_sample, params)
bin_size = max(get_param(params, 'bin_size_km', 1.5), 1e-3);
max_distance = min(get_param(params, 'max_dist_km', ...
    max(distance_km)), max(distance_km));
edges = 0:bin_size:(max_distance + bin_size);
bin = discretize(distance_km, edges);
valid_pair = isfinite(bin) & isfinite(covariance_values);
n_bin = numel(edges) - 1;
counts = accumarray(bin(valid_pair), 1, [n_bin, 1], @sum, 0);
sum_covariance = accumarray(bin(valid_pair), ...
    covariance_values(valid_pair), [n_bin, 1], @sum, 0);
empirical = sum_covariance ./ max(counts, 1);
centers = (edges(1:end-1) + edges(2:end))' / 2;
min_pairs = max(1, round(get_param(params, ...
    'exponential_min_bin_pairs', 30)));
positive = counts >= min_pairs & isfinite(empirical) & empirical > 0;
minimum_bins = max(3, round(get_param(params, ...
    'exponential_min_positive_bins', 5)));

sample_variance = max(var(residual_sample), 1e-6);
if nnz(positive) < minimum_bins
    C0 = max(sample_variance * 0.8, 1e-6);
    d0 = min(max(weighted_positive_range( ...
        distance_km, covariance_values, params), ...
        params.min_range_km), params.max_range_km);
    return;
end

d = centers(positive);
y = empirical(positive);
w = sqrt(counts(positive));
w = w ./ max(sum(w), eps);
C0_initial = min(max(max(y), 1e-6), 2 * sample_variance);
d0_initial = min(max(weighted_positive_range(d, y, params), ...
    params.min_range_km), params.max_range_km);
objective = @(z) exponential_fit_objective( ...
    z, d, y, w, sample_variance, params);
options = optimset('Display', 'off', 'MaxIter', ...
    get_param(params, 'exponential_fit_max_iterations', 160), ...
    'MaxFunEvals', 1000, 'TolX', 1e-6, 'TolFun', 1e-8);
z = fminsearch(objective, log([C0_initial, d0_initial]), options);
C0 = min(max(exp(z(1)), 1e-6), 2 * sample_variance);
d0 = min(max(exp(z(2)), params.min_range_km), params.max_range_km);
end

function value = exponential_fit_objective( ...
        z, distance, empirical, weight, sample_variance, params)
C0 = exp(z(1));
d0 = exp(z(2));
prediction = C0 .* exp(-distance ./ max(d0, 1e-9));
scale = max([max(empirical), sample_variance, 1e-6]);
residual = (prediction - empirical) ./ scale;
absolute = abs(residual);
huber_k = 1.5;
loss = 0.5 * residual .^ 2;
tail = absolute > huber_k;
loss(tail) = huber_k * absolute(tail) - 0.5 * huber_k ^ 2;
penalty = 0;
if d0 < params.min_range_km
    penalty = penalty + 10 * (log(params.min_range_km / d0)) ^ 2;
elseif d0 > params.max_range_km
    penalty = penalty + 10 * (log(d0 / params.max_range_km)) ^ 2;
end
if C0 > 2 * sample_variance
    penalty = penalty + 10 * (log(C0 / (2 * sample_variance))) ^ 2;
end
value = sum(weight .* loss) + penalty;
end

function range = weighted_positive_range(distance, covariance_values, params)
positive = isfinite(distance) & isfinite(covariance_values) & ...
    covariance_values > 0 & distance > 0;
if ~any(positive)
    range = 0.5 * (params.min_range_km + params.max_range_km);
    return;
end
w = covariance_values(positive);
range = sum(distance(positive) .* w) / max(sum(w), eps);
end

function shared_cache = prepare_shared_sample(coord_km, shared_cache, target_size, max_dist_km)
n_pts = size(coord_km, 1);
cache_ok = isstruct(shared_cache) && ...
    isfield(shared_cache, 'cache_format') && ...
    strcmp(shared_cache.cache_format, 'compact_pairs_v1') && ...
    isfield(shared_cache, 'coord_hash') && ...
    isfield(shared_cache, 'coord_all') && ...
    isequaln(shared_cache.coord_all,coord_km) && ...
    isfield(shared_cache, 'target_size') && ...
    shared_cache.target_size == target_size && ...
    isfield(shared_cache, 'max_dist_km') && ...
    shared_cache.max_dist_km == max_dist_km;
if cache_ok
    shared_cache.reused = true;
    return;
end

sample_idx = select_mixed_sample_indices((1:n_pts)', target_size);
coord_sample = coord_km(sample_idx, :);
[pair_i, pair_j, pair_dx, pair_dy, pair_d] = ...
    build_compact_pair_geometry(coord_sample, max_dist_km);

shared_cache = struct( ...
    'cache_format', 'compact_pairs_v1', ...
    'sample_idx', sample_idx, ...
    'coord_sample', coord_sample, ...
    'pair_i', pair_i, ...
    'pair_j', pair_j, ...
    'pair_dx', pair_dx, ...
    'pair_dy', pair_dy, ...
    'pair_d', pair_d, ...
    'pair_count', numel(pair_d), ...
    'target_size', target_size, ...
    'max_dist_km', max_dist_km, ...
    'coord_all', coord_km, 'coord_hash', simple_coord_hash(coord_km), ...
    'reused', false);
end

function [pair_i, pair_j, pair_dx, pair_dy, pair_d] = ...
    build_compact_pair_geometry(coord, max_dist_km)
% Generate pairs in the same column-major order as
% triu(mask,1)(:) so downstream summation order is unchanged.
n = size(coord, 1);
pair_count = 0;
for j = 2:n
    ii = (1:j-1)';
    dx = abs(coord(ii, 1) - coord(j, 1));
    dy = abs(coord(ii, 2) - coord(j, 2));
    d = hypot(dx, dy);
    pair_count = pair_count + nnz(d > 0 & d <= max_dist_km);
end

pair_i = zeros(pair_count, 1, 'uint32');
pair_j = zeros(pair_count, 1, 'uint32');
pair_dx = zeros(pair_count, 1);
pair_dy = zeros(pair_count, 1);
pair_d = zeros(pair_count, 1);
cursor = 0;
for j = 2:n
    ii = (1:j-1)';
    dx = abs(coord(ii, 1) - coord(j, 1));
    dy = abs(coord(ii, 2) - coord(j, 2));
    d = hypot(dx, dy);
    keep = d > 0 & d <= max_dist_km;
    count = nnz(keep);
    if count == 0
        continue;
    end
    target = cursor + (1:count);
    selected_i = ii(keep);
    pair_i(target) = uint32(selected_i);
    pair_j(target) = uint32(j);
    pair_dx(target) = dx(keep);
    pair_dy(target) = dy(keep);
    pair_d(target) = d(keep);
    cursor = cursor + count;
end
end

function [d_vec, dx_vec, dy_vec, cv_vec, rs] = ...
    shared_statistics_from_cache(r, shared_cache, source_mask)
d_vec = [];
dx_vec = [];
dy_vec = [];
cv_vec = [];
rs = [];
if isempty(shared_cache) || ~isfield(shared_cache, 'sample_idx')
    return;
end

sample_vals = r(shared_cache.sample_idx);
valid_sample = isfinite(sample_vals) & ...
    source_mask(shared_cache.sample_idx);
if sum(valid_sample) < 40
    return;
end

median_value = median(sample_vals(valid_sample));
centered = sample_vals;
centered(valid_sample) = centered(valid_sample) - median_value;
rs = centered(valid_sample);
pair_valid = valid_sample(shared_cache.pair_i) & ...
    valid_sample(shared_cache.pair_j);
if nnz(pair_valid) < 80
    d_vec = [];
    return;
end
d_vec = shared_cache.pair_d(pair_valid);
dx_vec = shared_cache.pair_dx(pair_valid);
dy_vec = shared_cache.pair_dy(pair_valid);
cv_vec = centered(shared_cache.pair_i(pair_valid)) .* ...
    centered(shared_cache.pair_j(pair_valid));
end

function [d_vec, dx_vec, dy_vec, cv_vec, rs] = local_statistics_from_valid(coord_valid, r_valid, params)
rs = r_valid - median(r_valid);
target_size = min(get_param(params, 'n_sample_cov', size(coord_valid, 1)), size(coord_valid, 1));
idx = select_mixed_sample_indices((1:size(coord_valid, 1))', target_size);
cs = coord_valid(idx, :);
rs = rs(idx);
dx = abs(cs(:, 1) - cs(:, 1)');
dy = abs(cs(:, 2) - cs(:, 2)');
d = hypot(dx, dy);
cv = rs * rs';
mask = triu(true(size(d)), 1) & d > 0 & d <= params.max_dist_km;
if nnz(mask) < 80
    d_vec = [];
    dx_vec = [];
    dy_vec = [];
    cv_vec = [];
    rs = [];
    return;
end
d_vec = d(mask);
dx_vec = dx(mask);
dy_vec = dy(mask);
cv_vec = cv(mask);
end

function idx = select_mixed_sample_indices(pool_idx, target_size)
n = numel(pool_idx);
if n <= target_size
    idx = pool_idx(:);
    return;
end

base = round(linspace(1, n, target_size))';
stride = max(floor(n / max(target_size, 1) / 3), 1);
jitter = mod((1:target_size)' * 37, 2 * stride + 1) - stride;
pos = min(max(base + jitter, 1), n);
pos = unique(pos, 'stable');

if numel(pos) < target_size
    fill_pos = setdiff(base, pos, 'stable');
    pos = [pos; fill_pos(1:min(numel(fill_pos), target_size - numel(pos)))];
end
pos = pos(1:min(target_size, numel(pos)));
idx = pool_idx(pos);
end

function h = simple_coord_hash(coord_km)
sample = coord_km(1:min(size(coord_km, 1), 25), :);
h = round(sum(sample(:) .* (1:numel(sample))') * 1e3);
end

function n = get_cache_sample_size(shared_cache)
if isstruct(shared_cache) && isfield(shared_cache, 'sample_idx')
    n = numel(shared_cache.sample_idx);
else
    n = 0;
end
end

function n = get_cache_pair_count(shared_cache)
if isstruct(shared_cache) && isfield(shared_cache, 'pair_count')
    n = shared_cache.pair_count;
else
    n = 0;
end
end

function tf = should_use_parallel_covariance(params, n_ifg)
tf = get_param(params, 'parallel_enable', false) && ...
    n_ifg >= get_param(params, 'parallel_min_ifgs', 16) && ...
    license('test', 'Distrib_Computing_Toolbox');
if ~tf
    return;
end
try
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('threads');
    end
    % The compact geometry can contain tens of millions of pairs. Only a
    % thread pool shares it without per-worker process copies. An explicit
    % execution-only override permits isolated workers when memory allows.
    tf = contains(class(pool), 'ThreadPool') || ...
        get_param(params,'parallel_allow_process',false);
catch exception
    warning('STAR_KF:CovarianceParallelUnavailable', ...
        'Parallel covariance unavailable; using serial mode: %s', ...
        exception.message);
    tf = false;
end
end

function ls = local_scale(coord_km, resid_ifg_rad, strength)
pv = var(resid_ifg_rad, 0, 2, 'omitnan');
pv(~isfinite(pv)) = median(pv(isfinite(pv)));
pvn = (pv - min(pv)) / max(max(pv) - min(pv), 1e-9);
center = mean(coord_km, 1, 'omitnan');
d = sqrt(sum((coord_km - center).^2, 2));
d = (d - min(d)) / max(max(d) - min(d), 1e-9);
ls = 1 + strength * (0.6 * pvn + 0.4 * d - 0.5);
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end

function variant = normalize_model_variant(params)
variant = lower(strtrim(char(get_param(params, 'model_variant', ...
    'dual_matern_anisotropic_nonstationary'))));
valid = {'single_exponential_isotropic_stationary', ...
    'single_matern_anisotropic_stationary', ...
    'dual_matern_isotropic_stationary', ...
    'dual_matern_anisotropic_stationary', ...
    'dual_matern_anisotropic_nonstationary'};
if ~ismember(variant, valid)
    error('Unknown spatial covariance model variant: %s', variant);
end
end

function tf = is_stationary_model(params)
variant = normalize_model_variant(params);
tf = ismember(variant, {'single_exponential_isotropic_stationary', ...
    'single_matern_anisotropic_stationary', ...
    'dual_matern_isotropic_stationary','dual_matern_anisotropic_stationary'});
end
