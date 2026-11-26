function [data, truth] = make_small_synthetic_case(scenario, seed)
%MAKE_SMALL_SYNTHETIC_CASE Deterministic compact STAR-KF test problem.

if nargin < 1 || isempty(scenario), scenario = 'mixed'; end
if nargin < 2 || isempty(seed), seed = 20260805; end
previous_rng = rng;
cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(seed, 'twister');

n_side = 8; % fixed common sample needs at least the existing 40-source statistics support
[X, Y] = meshgrid(linspace(0, 10, n_side));
coord_km = [X(:), Y(:)];
n_pts = size(coord_km, 1);
n_slc = 10;
i_master = 4;
acq_times = (0:(n_slc - 1))' .* 24;
t = (acq_times - acq_times(i_master)) ./ ...
    max(acq_times(end) - acq_times(1), 1);

g1 = exp(-((coord_km(:, 1) - 3.2) .^ 2 / 5 + ...
    (coord_km(:, 2) - 6.5) .^ 2 / 8));
g2 = exp(-((coord_km(:, 1) - 7.2) .^ 2 / 6 + ...
    (coord_km(:, 2) - 3.0) .^ 2 / 5));
linear = -18 .* g1 .* t';
acceleration = -9 .* g2 .* sign(t') .* abs(t') .^ 1.7;
seasonal = 3.5 .* (0.6 .* g1 + 0.4 .* g2) .* ...
    (sin(2 * pi .* acq_times' ./ 192) - ...
     sin(2 * pi .* acq_times(i_master) ./ 192));
jump_profile = double(acq_times >= acq_times(7));
jump_profile = jump_profile - jump_profile(i_master);
jump = -5 .* g2 .* jump_profile';
rebound_profile = -12 .* min(max((acq_times - acq_times(i_master)) ./ 72, 0), 1) + ...
    18 .* min(max((acq_times - acq_times(7)) ./ 72, 0), 1);
rebound_profile = rebound_profile - rebound_profile(i_master);
rebound = g1 .* rebound_profile';

switch lower(char(scenario))
    case 'pure_deformation'
        deformation_mm = linear + acceleration + seasonal + jump;
        turbulence_amplitude = 0;
    case 'pure_turbulence'
        deformation_mm = zeros(n_pts, n_slc);
        turbulence_amplitude = 6.0;
    case 'nonmonotonic'
        deformation_mm = rebound + 0.6 .* seasonal;
        turbulence_amplitude = 2.0;
    otherwise
        deformation_mm = linear + acceleration + seasonal + jump;
        turbulence_amplitude = 4.0;
end

mode1 = sin(2 * pi .* coord_km(:, 1) ./ 12) .* ...
    cos(2 * pi .* coord_km(:, 2) ./ 15);
mode2 = cos(2 * pi .* (coord_km(:, 1) + coord_km(:, 2)) ./ 18);
mode3 = sin(2 * pi .* coord_km(:, 2) ./ 9);
coeff = randn(3, n_slc);
turbulence_mm = turbulence_amplitude .* ...
    (0.50 .* mode1 .* coeff(1, :) + ...
     0.30 .* mode2 .* coeff(2, :) + ...
     0.20 .* mode3 .* coeff(3, :));
turbulence_mm = rebase_to_reference_epoch(turbulence_mm, i_master);
deformation_mm = rebase_to_reference_epoch(deformation_mm, i_master);

noise_std_mm = 0.20;
noise_mm = noise_std_mm .* randn(n_pts, n_slc);
noise_mm = rebase_to_reference_epoch(noise_mm, i_master);
observed_sequence_mm = deformation_mm + turbulence_mm + noise_mm;

ifg_pairs = zeros(0, 2);
for i = 1:n_slc
    for j = (i + 1):min(i + 3, n_slc)
        ifg_pairs(end + 1, :) = [i, j]; %#ok<AGROW>
    end
end
scale_rad2mm = -(0.0554657595313821 / (4 * pi)) * 1000;
phi_mat = sequence_to_ifg( ...
    observed_sequence_mm, ifg_pairs, 1 / scale_rad2mm);

data = struct( ...
    'phi_mat', phi_mat, ...
    'coord_km', coord_km, ...
    'z_d_raw', observed_sequence_mm, ...
    'acq_times', acq_times, ...
    'ifg_times', [acq_times(ifg_pairs(:, 1)), ...
        acq_times(ifg_pairs(:, 2))], ...
    'ifg_pairs', ifg_pairs, ...
    'scale_rad2mm', scale_rad2mm, ...
    'n_pts', n_pts, ...
    'n_ifg', size(ifg_pairs, 1), ...
    'n_slc', n_slc, ...
    'i_master', i_master);
truth = struct( ...
    'phi_def_mm', deformation_mm, ...
    'phi_turb_mm', turbulence_mm, ...
    'phi_noise_mm', noise_mm, ...
    'phi_turb_ifg_rad', sequence_to_ifg( ...
        turbulence_mm, ifg_pairs, 1 / scale_rad2mm));
data.truth = truth;
end

