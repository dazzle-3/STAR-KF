function data = load_experiment_data(input_dir, cfg)
input_path = fullfile(input_dir, cfg.input_file);
kf_path = fullfile(input_dir, cfg.kf_input_file);
log_progress('DATA', 'Loading interferogram data: %s', input_path);
S1 = load(input_path);
log_progress('DATA', 'Loading deformation time-series data: %s', kf_path);
S2 = load(kf_path);
phi_mat = double(resolve_field(S1, {'ph_mat', 'ph_all'}));
xy = double(resolve_field(S1, {'xy'}));
if size(xy, 2) >= 3
    coord_km = xy(:, 2:3) / 1000.0;
else
    coord_km = xy / 1000.0;
end
z_d_raw = double(resolve_field(S2, {'ph_mm_full'}));
acq_times = double(resolve_field(S2, {'day'}));
acq_times = acq_times(:);
ifg_times = double(resolve_field(S2, {'ifgday'}));
if size(ifg_times, 2) ~= 2
    ifg_times = reshape(ifg_times, [], 2);
end
ifg_pairs = ifgday_to_pairs_idx(acq_times, ifg_times);
validate_dimensions(phi_mat, coord_km, z_d_raw, acq_times, ifg_pairs);
unit_audit = validate_phase_timeseries_units( ...
    phi_mat, z_d_raw, ifg_pairs, cfg);

data = struct('phi_mat', phi_mat, 'coord_km', coord_km, 'z_d_raw', z_d_raw, ...
    'acq_times', acq_times, 'ifg_times', ifg_times, 'ifg_pairs', ifg_pairs, ...
    'scale_rad2mm', cfg.scale_rad2mm, 'n_pts', size(phi_mat, 1), ...
    'n_ifg', size(phi_mat, 2), 'n_slc', size(z_d_raw, 2), ...
    'i_master', min(max(round(cfg.i_master), 1), size(z_d_raw, 2)), ...
    'unit_audit', unit_audit);
log_progress('DATA', 'Validated dimensions: %d points, %d IFGs, %d SLCs', data.n_pts, data.n_ifg, data.n_slc);

truth = load_optional_truth(input_dir, cfg, ifg_pairs);
if ~isempty(truth)
    validate_truth(truth, data);
    data.truth = truth;
    log_progress('DATA', 'Ground truth is available for quantitative evaluation');
end

function audit = validate_phase_timeseries_units( ...
    phi_mat, z_d_raw, ifg_pairs, cfg)
audit = struct( ...
    'enabled', false, ...
    'internal_unit', 'unspecified', ...
    'central_slope_phase_on_timeseries', NaN, ...
    'central_correlation', NaN, ...
    'n_samples', 0, ...
    'within_nominal_tolerance', false, ...
    'status', 'not_run');
if ~isfield(cfg, 'units') || ~isstruct(cfg.units)
    return;
end
audit.internal_unit = char(get_param(cfg.units, ...
    'internal_epoch', 'unspecified'));
if ~get_param(cfg.units, 'validate_phase_timeseries', false)
    return;
end

n_pts = size(phi_mat, 1);
n_ifg = size(phi_mat, 2);
max_pts = max(round(get_param(cfg.units, ...
    'validation_max_points', 4000)), 100);
max_ifgs = max(round(get_param(cfg.units, ...
    'validation_max_ifgs', 12)), 3);
point_idx = unique(round(linspace(1, n_pts, min(max_pts, n_pts))));
ifg_idx = unique(round(linspace(1, n_ifg, min(max_ifgs, n_ifg))));

phase_mm = double(phi_mat(point_idx, ifg_idx)) .* ...
    cfg.scale_rad2mm;
ts_ifg_mm = nan(numel(point_idx), numel(ifg_idx));
for k = 1:numel(ifg_idx)
    pair = ifg_pairs(ifg_idx(k), :);
    ts_ifg_mm(:, k) = z_d_raw(point_idx, pair(2)) - ...
        z_d_raw(point_idx, pair(1));
end

x = ts_ifg_mm(:);
y = phase_mm(:);
valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);
if numel(x) < 100
    error('Unit validation has only %d finite samples.', numel(x));
end

abs_x = abs(x);
abs_y = abs(y);
keep = abs_x <= prctile(abs_x, 95) & ...
    abs_y <= prctile(abs_y, 95);
x = x(keep);
y = y(keep);
slope = sum(x .* y, 'omitnan') / ...
    max(sum(x .^ 2, 'omitnan'), 1e-12);
C = corrcoef(x, y);
corr_value = C(1, 2);
tolerance = get_param(cfg.units, ...
    'validation_slope_tolerance', 0.08);
hard_bounds = get_param(cfg.units, ...
    'validation_slope_hard_bounds', [0.50, 1.50]);
hard_bounds = sort(double(hard_bounds(:)'));
if numel(hard_bounds) ~= 2 || any(~isfinite(hard_bounds))
    hard_bounds = [0.50, 1.50];
end
if ~isfinite(slope) || slope < hard_bounds(1) || slope > hard_bounds(2)
    error(['Phase/time-series unit validation failed: central slope ', ...
        'phase_mm on diagnostic raw time-series is %.4f, outside hard ', ...
        'LOS-mm bounds [%.3f, %.3f].'], slope, hard_bounds(1), ...
        hard_bounds(2));
end

audit.enabled = true;
audit.central_slope_phase_on_timeseries = slope;
audit.central_correlation = corr_value;
audit.n_samples = numel(x);
audit.within_nominal_tolerance = abs(slope - 1) <= tolerance;
if audit.within_nominal_tolerance
    audit.status = 'nominal';
    log_progress('DATA', ...
        ['Unit audit passed: internal=%s, phase/diagnostic-raw slope=', ...
         '%.4f, correlation=%.4f, samples=%d'], ...
        audit.internal_unit, slope, corr_value, audit.n_samples);
else
    audit.status = 'diagnostic_raw_amplitude_mismatch';
    log_progress('DATA', ...
        ['Unit audit diagnostic: phase/diagnostic-raw slope=%.4f is ', ...
         'outside nominal 1 +/- %.4f but within hard LOS-mm bounds; ', ...
         'data.z_d_raw is not used by the formal observation chain'], ...
        slope, tolerance);
end
end
end

function value = resolve_field(S, candidates)
for i = 1:numel(candidates)
    name = candidates{i};
    if isfield(S, name)
        value = S.(name);
        return;
    end
end
error('Missing required field. Expected one of: %s.', strjoin(candidates, ', '));
end

function truth = load_optional_truth(input_dir, cfg, ifg_pairs)
truth = struct();
truth_path = fullfile(input_dir, cfg.truth_file);
if ~isfile(truth_path)
    log_progress('DATA', 'Ground truth not found at %s; using reference-free evaluation only', truth_path);
    truth = [];
    return;
end

log_progress('DATA', 'Loading ground truth: %s', truth_path);
S = load(truth_path);
gt = extract_truth_struct(S);

if isfield(gt, 'phi_true_rad')
    truth.phi_def_mm = double(gt.phi_true_rad) * cfg.scale_rad2mm;
end
if isfield(gt, 'phi_true_ifg_rad')
    truth.phi_def_ifg_rad = double(gt.phi_true_ifg_rad);
end
if isfield(gt, 'phi_atm_rad')
    truth.phi_turb_mm = double(gt.phi_atm_rad) * cfg.scale_rad2mm;
    truth.phi_turb_ifg_rad = sequence_to_ifg(truth.phi_turb_mm, ifg_pairs, 1 / cfg.scale_rad2mm);
end
if isfield(gt, 'phi_decorr_rad')
    truth.phi_noise_mm = double(gt.phi_decorr_rad) * cfg.scale_rad2mm;
end
if isfield(gt, 'weather_types')
    truth.weather_type = double(gt.weather_types(:)');
elseif isfield(gt, 'weather_type')
    truth.weather_type = double(gt.weather_type(:)');
end
if isfield(gt, 'coord_km')
    truth.coord_km = double(gt.coord_km);
end
if isfield(gt, 'acq_dates')
    truth.acq_dates = gt.acq_dates;
end
truth.raw = gt;

if isempty(fieldnames(truth))
    truth = [];
end
end

function gt = extract_truth_struct(S)
if isfield(S, 'ground_truth') && isstruct(S.ground_truth)
    gt = S.ground_truth;
    return;
end

names = fieldnames(S);
if numel(names) == 1 && isstruct(S.(names{1}))
    gt = S.(names{1});
    return;
end

gt = S;
end

function validate_dimensions(phi_mat, coord_km, z_d_raw, acq_times, ifg_pairs)
if size(phi_mat, 1) ~= size(coord_km, 1)
    error('input_data coordinate rows (%d) do not match interferogram rows (%d).', size(coord_km, 1), size(phi_mat, 1));
end
if size(phi_mat, 1) ~= size(z_d_raw, 1)
    error('ph_mm_full rows (%d) do not match input_data rows (%d).', size(z_d_raw, 1), size(phi_mat, 1));
end
if size(phi_mat, 2) ~= size(ifg_pairs, 1)
    error('Interferogram count mismatch: ph_mat has %d columns but ifgday resolves to %d pairs.', size(phi_mat, 2), size(ifg_pairs, 1));
end
if size(z_d_raw, 2) ~= numel(acq_times)
    error('SLC count mismatch: ph_mm_full has %d columns but day has %d timestamps.', size(z_d_raw, 2), numel(acq_times));
end
end

function validate_truth(truth, data)
if isfield(truth, 'phi_def_mm') && ~isequal(size(truth.phi_def_mm), [data.n_pts, data.n_slc])
    error('ground_truth deformation size must be %d x %d.', data.n_pts, data.n_slc);
end
if isfield(truth, 'phi_turb_mm') && ~isequal(size(truth.phi_turb_mm), [data.n_pts, data.n_slc])
    error('ground_truth turbulence size must be %d x %d.', data.n_pts, data.n_slc);
end
if isfield(truth, 'phi_def_ifg_rad') && ~isequal(size(truth.phi_def_ifg_rad), [data.n_pts, data.n_ifg])
    error('ground_truth interferogram truth size must be %d x %d.', data.n_pts, data.n_ifg);
end
if isfield(truth, 'phi_turb_ifg_rad') && ~isequal(size(truth.phi_turb_ifg_rad), [data.n_pts, data.n_ifg])
    error('ground_truth atmospheric IFG size must be %d x %d.', data.n_pts, data.n_ifg);
end
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
