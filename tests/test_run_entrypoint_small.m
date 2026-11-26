function tests = test_run_entrypoint_small
tests = functiontests(localfunctions);
end

function setupOnce(~)
project = fileparts(fileparts(mfilename('fullpath')));
addpath(project, fullfile(project, 'modules'), ...
    fullfile(project, 'utils'), fullfile(project, 'evaluation'), ...
    fullfile(project, 'tests'));
end

function testPublicEntrypointAndSavedContract(testCase)
project = fileparts(fileparts(mfilename('fullpath')));
[data, ~] = make_small_synthetic_case('mixed', 41);
fixture_root = tempname(fullfile(project, 'tests'));
input_dir = fullfile(fixture_root, 'input');
output_dir = fullfile(fixture_root, 'output');
mkdir(input_dir);
cleanup = onCleanup(@() remove_fixture(fixture_root)); %#ok<NASGU>

ph_mat = data.phi_mat; %#ok<NASGU>
xy = [(1:data.n_pts)', data.coord_km .* 1000]; %#ok<NASGU>
save(fullfile(input_dir, 'input_data.mat'), 'ph_mat', 'xy');
ph_mm_full = data.z_d_raw; %#ok<NASGU>
day = data.acq_times; %#ok<NASGU>
ifgday = data.ifg_times; %#ok<NASGU>
save(fullfile(input_dir, 'ph_mm_full.mat'), ...
    'ph_mm_full', 'day', 'ifgday');

cfg = small_test_config(1);
outputs = run_star_kf_v4(input_dir, output_dir, cfg);
verifyTrue(testCase, isfile(outputs.result_path));
verifyTrue(testCase, ...
    isfile(fullfile(output_dir, 'experiment_setup_summary.md')));
verifyTrue(testCase, ...
    isfile(fullfile(output_dir, 'quantitative_metrics_summary.md')));
verifyEqual(testCase, outputs.result.deformation_joint_final, ...
    outputs.result.phi_seq_corrected_mm, 'AbsTol', 0);
verifyFalse(testCase, isfield(outputs.result, 'iterback_history'));
verifyFalse(testCase, ...
    outputs.result.output_contract.post_kriging_deformation_recovery);
end

function remove_fixture(path_value)
if isfolder(path_value)
    rmdir(path_value, 's');
end
end

