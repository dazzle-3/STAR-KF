function results = run_all_tests()
%RUN_ALL_TESTS Run repeatable formal-flow regression tests.

project = fileparts(fileparts(mfilename('fullpath')));
addpath(project, fullfile(project, 'modules'), ...
    fullfile(project, 'utils'), fullfile(project, 'evaluation'), ...
    fullfile(project, 'tests'));
suite = testsuite(fullfile(project, 'tests'), ...
    'IncludeSubfolders', false);
results = run(suite);
fprintf('STAR-KF tests: %d passed, %d failed, %d incomplete\n', ...
    nnz([results.Passed]), nnz([results.Failed]), ...
    nnz([results.Incomplete]));
evidence=fullfile(project,'validation','tests');
if ~isfolder(evidence), mkdir(evidence); end
    summary=struct('names',{cellstr(string({results.Name}))}, ...
        'passed',[results.Passed],'failed',[results.Failed], ...
        'incomplete',[results.Incomplete],'duration_seconds',[results.Duration]);
    fid=fopen(fullfile(evidence,'current_test_summary.json'),'w','n','UTF-8');
    fprintf(fid,'%s',jsonencode(summary,'PrettyPrint',true)); fclose(fid);
    save(fullfile(evidence,'all_test_results.mat'),'results');
assertSuccess(results);
end

