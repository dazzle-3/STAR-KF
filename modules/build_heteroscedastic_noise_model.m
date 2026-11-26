function model = build_heteroscedastic_noise_model( ...
    coord_km, resid_ifg_rad, opt)
%BUILD_HETEROSCEDASTIC_NOISE_MODEL Point-noise model for Kriging.

if ~get_param(opt, 'hetero_kriging_nugget', false)
    model = [];
    return;
end
method = lower(char(get_param( ...
    opt, 'hetero_score_method', 'spatial_highpass_ifg')));
switch method
    case {'spatial_highpass_ifg', 'local_highpass', 'ifg_highpass'}
        model = spatial_highpass_model(coord_km, resid_ifg_rad, opt);
    case {'point_variance', 'legacy'}
        model = point_variance_model(resid_ifg_rad, opt);
    otherwise
        error('Unknown heteroscedastic score method: %s.', method);
end
end

function model = spatial_highpass_model(coord_km, resid_ifg_rad, opt)
[n_pts, n_ifg] = size(resid_ifg_rad);
cell_km = max(get_param(opt, 'hetero_local_cell_km', 8.0), 0.5);
min_cell_pts = max(round(get_param(opt, ...
    'hetero_min_cell_points', 40)), 3);
active_fraction = min(max(get_param(opt, ...
    'hetero_active_fraction', 0.10), 0), 1);
strength = max(get_param(opt, 'hetero_score_strength', 0.25), 0);
score_cap = max(get_param(opt, 'hetero_score_cap', 0.50), 0);
variance_weight = min(max(get_param(opt, ...
    'hetero_variance_weight', 0.20), 0), 1);

[cell_id, n_cells, cell_counts] = spatial_cells(coord_km, cell_km);
global_median = median(resid_ifg_rad, 1, 'omitnan');
cell_median = repmat(global_median, n_cells, 1);
[sorted_cell, order] = sort(cell_id);
breaks = [1; find(diff(sorted_cell) ~= 0) + 1; n_pts + 1];
for g = 1:(numel(breaks) - 1)
    c = sorted_cell(breaks(g));
    idx = order(breaks(g):(breaks(g + 1) - 1));
    if numel(idx) < min_cell_pts, continue; end
    med = median(resid_ifg_rad(idx, :), 1, 'omitnan');
    missing = ~isfinite(med);
    med(missing) = global_median(missing);
    cell_median(c, :) = med;
end

scale = nan(1, n_ifg);
threshold = inf(1, n_ifg);
for k = 1:n_ifg
    local_median = cell_median(cell_id, k);
    anomaly = abs(resid_ifg_rad(:, k) - local_median);
    values = anomaly(isfinite(anomaly));
    if numel(values) < 8
        scale(k) = 1;
        continue;
    end
    center = median(values, 'omitnan');
    scale(k) = max([1.4826 * median(abs(values - center), 'omitnan'), ...
        iqr(values) / 1.349, center, 1e-6]);
    if active_fraction > 0
        threshold(k) = prctile(values, 100 * (1 - active_fraction));
    end
end
model = struct( ...
    'mode', 'spatial_highpass_ifg', ...
    'cell_id', uint32(cell_id), ...
    'cell_median', cell_median, ...
    'scale', scale, ...
    'threshold', threshold, ...
    'strength', strength, ...
    'score_cap', score_cap, ...
    'variance_weight', variance_weight, ...
    'active_fraction', active_fraction, ...
    'cell_km', cell_km, ...
    'cell_counts', uint32(cell_counts), ...
    'n_pts', n_pts, ...
    'n_ifg', n_ifg);
end

function model = point_variance_model(resid_ifg_rad, opt)
variance = var(resid_ifg_rad, 0, 2, 'omitnan');
valid = isfinite(variance);
score = zeros(size(variance));
if any(valid)
    center = median(variance(valid), 'omitnan');
    spread = max([iqr(variance(valid)), ...
        1.4826 * median(abs(variance(valid) - center), 'omitnan'), ...
        center, 1e-6]);
    score = max((variance - center) ./ spread, 0);
end
score(~isfinite(score)) = 0;
score = min(score .* get_param(opt, ...
    'hetero_score_strength', 0.25), ...
    get_param(opt, 'hetero_score_cap', 0.50));
model = struct( ...
    'mode', 'static_score', ...
    'score', score, ...
    'variance_weight', min(max(get_param(opt, ...
        'hetero_variance_weight', 0.20), 0), 1), ...
    'n_pts', size(resid_ifg_rad, 1), ...
    'n_ifg', size(resid_ifg_rad, 2));
end

function [cell_id, n_cells, cell_counts] = spatial_cells(coord_km, cell_km)
x = coord_km(:, 1);
y = coord_km(:, 2);
nx = max(ceil((max(x) - min(x)) / cell_km), 1);
ny = max(ceil((max(y) - min(y)) / cell_km), 1);
x_edges = linspace(min(x), max(x), nx + 1);
y_edges = linspace(min(y), max(y), ny + 1);
[~, ~, ix] = histcounts(x, x_edges);
[~, ~, iy] = histcounts(y, y_edges);
ix(ix == 0) = nx;
iy(iy == 0) = ny;
ix = min(max(ix, 1), nx);
iy = min(max(iy, 1), ny);
cell_id = sub2ind([nx, ny], ix, iy);
n_cells = nx * ny;
cell_counts = accumarray(cell_id, 1, [n_cells, 1]);
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end

