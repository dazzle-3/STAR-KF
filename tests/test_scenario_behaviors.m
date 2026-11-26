function tests=test_scenario_behaviors
tests=functiontests(localfunctions);
end
function testScenarioAlgebraicContracts(testCase)
% Old estimator's empirical RMSE/transfer gates are retained in baseline_before.zip.
% Current method is checked against its mathematical contract; no truth selects settings.
scenarios={'pure_deformation','pure_turbulence','mixed','nonmonotonic'};
for k=1:numel(scenarios)
    [data,~]=make_small_synthetic_case(scenarios{k},30+k); cfg=small_test_config();
    with_truth=data; data=rmfield(data,'truth');
    result=joint_inversion(data,cfg);
    verifyTrue(testCase,all(isfinite(result.deformation_joint_final(:))));
    verifyTrue(testCase,all(isfinite(result.P_joint_final_mm2(:))));
    [B,~,~,~]=network_incidence(data.ifg_pairs,data.n_slc,data.i_master);
    verifyEqual(testCase,result.turbulence_final,result.atmosphere_epoch_final_rad*B','AbsTol',1e-12);
    verifyEqual(testCase,result.residual_final,data.phi_mat-sequence_to_ifg(result.deformation_joint_final,data.ifg_pairs,1/data.scale_rad2mm)-result.turbulence_final,'AbsTol',1e-12);
    verifyGreaterThanOrEqual(testCase,numel(result.iterations),4);
    verifyLessThanOrEqual(testCase,numel(result.iterations),cfg.joint.n_outer_iters);
    verifyFalse(testCase,result.finalization.spatial_update_performed);
    verifyTrue(testCase,isfinite(result.finalization.d_final));
    for p=1:numel(result.finalization.deformation.operators)
        op=result.finalization.deformation.operators{p}; ix=result.finalization.deformation.operator_pixel_indices(p);
        A=op.A1+op.w*op.KF2.F2;
        verifyEqual(testCase,result.P_joint_final_mm2(ix,:),diag(A*op.SigmaY*A')','AbsTol',1e-9);
    end
    if k==3
        same=joint_inversion(with_truth,cfg);
        verifyEqual(testCase,same.deformation_joint_final,result.deformation_joint_final,'AbsTol',0);
    end
end
end
function testOutlierAndWorkingVarianceAccounting(testCase)
[data,~]=make_small_synthetic_case('mixed',34); cfg=small_test_config();
[Y,ob]=build_corrected_observation_sequence(data,zeros(size(data.phi_mat)),16*ones(1,data.n_slc),cfg);
Y(1:3:end,8)=Y(1:3:end,8)+60;
result=two_stage_deformation_estimation(Y,data,16*ones(1,data.n_slc),cfg,ob.operators);
verifyTrue(testCase,all(isfinite(result.D_joint_q(:))));
verifyLessThanOrEqual(testCase,max(result.weights.W2),.8);
vce=struct('sigma2_slc',.1*ones(1,data.n_slc),'known_noise_ifg_rad2',.02*ones(1,data.n_ifg));
vsp=2*ones(data.n_pts,data.n_ifg); E=3*ones(size(vsp)); P=5*ones(size(Y)); P(:,data.i_master)=0;
[R,c]=propagate_joint_uncertainty(vsp,vce,E,P,10*ones(1,data.n_slc),data,cfg,true(size(E)));
verifyEqual(testCase,c.ifg_kriging_latent_rad2,2*ones(1,data.n_ifg),'AbsTol',0);
verifyEqual(testCase,c.ifg_residual_positive_rad2,(9-.22-2)*ones(1,data.n_ifg),'AbsTol',1e-12);
verifyEqual(testCase,c.raw_target_mm2,c.slc_vce_mm2+c.slc_kriging_latent_mm2+c.slc_residual_positive_mm2+c.slc_deformation_participation_mm2,'AbsTol',1e-12);
verifyGreaterThan(testCase,R(data.i_master),cfg.uncertainty.r_floor_mm2);
end
