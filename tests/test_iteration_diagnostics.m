function tests=test_iteration_diagnostics
tests=functiontests(localfunctions);
end

function testContinuationRecordsCriterionWithoutChangingPrefix(testCase)
[data,truth]=make_small_synthetic_case('mixed',57);
data=rmfield(data,'truth'); cfg=small_test_config();
cfg.joint.n_outer_iters=6;
cfg.convergence.joint_state_tol=3; % fixture-only trigger: relative norm change is <=2.
% Omitted diagnostic settings preserve compatibility with older run configs.
cfg=rmfield(cfg,'diagnostics'); normal=joint_inversion(data,cfg);
cfg.diagnostics=struct('continue_after_convergence',true);
extended=joint_inversion(data,cfg);
verifyEqual(testCase,numel(normal.iterations),4);
verifyEqual(testCase,numel(extended.iterations),6);
verifyEqual(testCase,normal.criterion_first_satisfied_iteration,4);
verifyEqual(testCase,extended.criterion_first_satisfied_iteration,4);
verifyTrue(testCase,extended.criterion_ever_fulfilled);
verifyEqual(testCase,normal.termination_reason,'joint_state_criterion');
verifyEqual(testCase,extended.termination_reason,'diagnostic_iteration_limit');
verifyEqual(testCase,extended.runtime_profile.fresh_spatial_updates,6);
for q=1:4
    for field={'D_joint_q','tau_q','R_next','P_joint_q'}
        verifyEqual(testCase,normal.iterations{q}.(field{1}),extended.iterations{q}.(field{1}),'AbsTol',0);
    end
end
verifyFalse(testCase,extended.finalization.spatial_update_performed);
% Evaluation-only fixture with explicit background rows, created after solving.
evaluation_data=data; evaluation_data.truth=truth;
evaluation_data.truth.phi_def_mm(1:8,:)=0;
history=evaluate_iteration_history(evaluation_data,extended,[]);
verifyEqual(testCase,numel(history),12);
verifyEqual(testCase,history(end).stage,'final_if_stop');
verifyEqual(testCase,history(end).source,'actual_finalizer');
keep=setdiff(1:data.n_slc,data.i_master);
e=extended.deformation_joint_final-evaluation_data.truth.phi_def_mm;
verifyEqual(testCase,history(end).joint_rmse_mm,sqrt(mean(e(:,keep).^2,'all')),'AbsTol',1e-12);
e=extended.iterations{4}.D_joint_q-evaluation_data.truth.phi_def_mm;
verifyEqual(testCase,history(6).joint_rmse_mm,sqrt(mean(e(:,keep).^2,'all')),'AbsTol',1e-12);
end

function testRejectNonlogicalContinuation(testCase)
[data,~]=make_small_synthetic_case('mixed',58); cfg=small_test_config();
cfg.diagnostics.continue_after_convergence=1;
verifyError(testCase,@()joint_inversion(data,cfg),'STAR_KF:DiagnosticControl');
end
