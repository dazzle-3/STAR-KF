function tests = test_formal_flow_static
tests = functiontests(localfunctions);
end

function testNoPostKrigingRecoveryInFormalModules(testCase)
project = fileparts(fileparts(mfilename('fullpath')));
formal_files = [dir(fullfile(project, '*.m')); ...
    dir(fullfile(project, 'modules', '*.m')); ...
    dir(fullfile(project, 'utils', '*.m'))];
forbidden = {'iteration_leakage_backsubstitution', ...
    'iteration_leakage_recovery', 'deformation_leakage_recovery', ...
    'compose_recovered_v2', 'posterior_sequence_update', ...
    'apply_iteration_transfer_guard', 'apply_deformation_guard_if_enabled'};
for i = 1:numel(formal_files)
    text = fileread(fullfile(formal_files(i).folder, formal_files(i).name));
    for j = 1:numel(forbidden)
        verifyFalse(testCase, contains(text, forbidden{j}), ...
            sprintf('%s contains forbidden formal call %s', ...
            formal_files(i).name, forbidden{j}));
    end
end
end

function testLegacyIsOutsideFormalPath(testCase)
project = fileparts(fileparts(mfilename('fullpath')));
entry_text = fileread(fullfile(project, 'run_star_kf_v4.m'));
verifyFalse(testCase, contains(entry_text, 'addpath(genpath(root_dir))'));
verifyTrue(testCase, isfile(fullfile(project, 'legacy', 'archive', ...
    'iteration_leakage_backsubstitution.m')));
end

