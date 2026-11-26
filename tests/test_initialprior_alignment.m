function tests=test_initialprior_alignment
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'modules'),fullfile(root,'utils'),fullfile(root,'tests'));
end
function testNetworkCovarianceAndMissingMarginal(testCase)
cfg=star_kf_v4_config(); n=4; ref=3;
pairs=[1 2;2 3;3 4;1 3;2 4]; R=[2,3,40,7];
model=build_ifg_covariance_model(pairs,n,ref,R,.2,1e-4);
[B,J,Br]=network_incidence(pairs,n,ref);
verifyEqual(testCase,model.Dyy,B*diag(R)*B'+.2*eye(5),'AbsTol',1e-12);
phi=[1,2,4,3,6;1,NaN,4,3,6];
[Y,o]=recover_sequence_phase_weighted(phi,n,pairs,ref,model,1e-6);
for p=1:2
    valid=isfinite(phi(p,:)); P=(model.Dyy(valid,valid)+model.eps_reg*eye(nnz(valid)))\eye(nnz(valid));
    G=J*((Br(valid,:)'*P*Br(valid,:)+1e-6*eye(n-1))\(Br(valid,:)'*P));
    ix=o.pattern_index(p); actual=o.Gphase(:,valid,ix);
    verifyEqual(testCase,actual,G,'AbsTol',1e-10);
    verifyEqual(testCase,Y(p,:)',G*phi(p,valid)','AbsTol',1e-10);
    verifyEqual(testCase,o.SigmaPhase(:,:,ix),G*model.Dyy(valid,valid)*G','AbsTol',1e-10);
end
valid=isfinite(phi(2,:));
verifyGreaterThan(testCase,norm(model.Pyy(valid,valid)-inv(model.Dyy(valid,valid)+model.eps_reg*eye(nnz(valid))),'fro'),1e-3);
verifyEqual(testCase,Y(:,ref),zeros(2,1),'AbsTol',0);
verifyGreaterThan(testCase,abs(o.SigmaPhase(1,2,1)),0);
end
function testKF1FrozenOperatorAndSEA(testCase)
cfg=star_kf_v4_config(); t=0:24:240; n=numel(t); ref=4;
y=sin(t/90)+.01*t; y=y-y(ref); y(8)=200;
names={'cv','ca','seasonal'};
for m=1:3
    c=build_kalman_model_cache(t,cfg.kalman,names{m});
    r=kalman_filter(y,t,5*ones(1,n),cfg.kalman,names{m},ref,c);
    K=build_trajectory_prior(c,cfg.kalman.q_acc,[]);
    ids=find(r.accepted); S=zeros(numel(ids),n);
    for j=1:numel(ids), S(j,ids(j))=1; end
    T=eye(n); T(:,ref)=T(:,ref)-1;
    A=T*K(:,ids)/(K(ids,ids)+diag(5*r.inflation(ids)))*S;
    verifyEqual(testCase,r.A,A,'AbsTol',2e-9);
    verifyEqual(testCase,r.sequence_mm,(A*y')','AbsTol',2e-9);
    verifyEqual(testCase,r.step_loglike(ref),0,'AbsTol',0);
    verifyFalse(testCase,r.accepted(8));
    verifyLessThan(testCase,r.step_loglike(8),0);
    c0=c; c0.P0=zeros(c.dim); K0=build_trajectory_prior(c0,cfg.kalman.q_acc,[]);
    verifyGreaterThan(testCase,norm(K-K0,'fro'),1);
    verifyGreaterThanOrEqual(testCase,min(eig(build_trajectory_prior(c,cfg.kalman.q_acc,ref))),-1e-8);
end
sea=build_kalman_model_cache(t,cfg.kalman,'seasonal');
verifyEqual(testCase,sea.H,repmat([1,0,1,0],n,1),'AbsTol',0);
verifyEqual(testCase,sea.Q0(:,:,1),zeros(4),'AbsTol',0);
end
function testStudentDensityAndWeights(testCase)
cfg=star_kf_v4_config(); p=cfg.kalman; S0=3.7;
pdf=@(x) arrayfun(@(a) exp(student_predictive_score(a,S0,p)),x);
verifyEqual(testCase,integral(pdf,-Inf,Inf),1,'AbsTol',1e-8);
verifyEqual(testCase,integral(@(x)x.^2.*pdf(x),-Inf,Inf),S0,'AbsTol',1e-6);
[s,a,u]=student_predictive_score(2,S0,p);
nu=p.student_df; b2=(nu-2)*S0/nu;
verifyEqual(testCase,s,gammaln((nu+1)/2)-gammaln(nu/2)-.5*log(nu*pi*b2)-(nu+1)/2*log1p(4/(nu*b2)),'AbsTol',1e-12);
verifyEqual(testCase,a,max([1,(nu+u^2)/(nu+1),u/p.huber_k]),'AbsTol',0);
scores=[0 -1 -3 0;0 -2 -1 0;0 -4 -2 0];
[alpha,mu,xi]=predictive_model_weights(scores,p);
T=.01*ones(3); T(1:4:end)=.98;
verifyEqual(testCase,mu(:,1),T'*p.model_prior(:),'AbsTol',1e-12);
verifyEqual(testCase,alpha(:,end),mu(:,end),'AbsTol',0);
verifyEqual(testCase,sum(alpha,1),ones(1,4),'AbsTol',1e-12);
verifyEqual(testCase,alpha(:,2),xi(:,:,3)*alpha(:,3),'AbsTol',1e-12);
end
function testKF2WhiteningSequentialBatchAndScores(testCase)
rng(701); cfg=star_kf_v4_config(); n=9; ref=5; t=0:24:192;
T=eye(n); T(:,ref)=T(:,ref)-1;
S=T*diag(1:n)*T'; A1=.25*T; e=randn(1,n); e(ref)=0; e(8)=100;
names={'cv','ca','seasonal'}; Kd=cell(1,3);
for m=1:3
    c=build_kalman_model_cache(t,cfg.kalman_kf2,names{m});
    Kd{m}=build_trajectory_prior(c,cfg.kalman_kf2.q_acc,ref);
end
% One model deliberately has a rank-one PSD prior.
v=T*(1:n)'; Kd{1}=v*v';
o=correlated_residual_filter(e,A1,S,Kd,cfg.kalman_kf2,ref);
verifyLessThan(testCase,o.whitening_error,1e-10);
rejected=0;
for m=1:3
    r=o.models{m}; W=r.Wm; K=r.Kdelta;
    batch=K*W'/(W*K*W'+r.Gamma)*W;
    verifyEqual(testCase,r.A2,batch,'AbsTol',5e-10);
    verifyEqual(testCase,r.mean,batch*e','AbsTol',5e-10);
    post=K-K*W'/(W*K*W'+r.Gamma)*W*K;
    verifyEqual(testCase,r.posterior_diagnostic,post,'AbsTol',2e-9);
    verifyEqual(testCase,o.scores(m,ref),0,'AbsTol',0);
    keep=setdiff(1:n,ref); rejected=rejected+nnz(~r.accepted);
    verifyTrue(testCase,all(o.scores(m,keep(~r.accepted))~=0));
end
verifyGreaterThan(testCase,rejected,0);
verifyEqual(testCase,o.F2,o.A2*(eye(n)-A1),'AbsTol',0);
verifyEqual(testCase,o.C2,o.F2*S*o.F2','AbsTol',1e-10);
verifyError(testCase,@()correlated_residual_filter(e,eye(n),S,Kd,cfg.kalman_kf2,ref),'STAR_KF:KF2ReducedRank');
end
function testAcceptanceProjectionAndCrossCovariance(testCase)
cfg=star_kf_v4_config(); n=7; ref=3; T=eye(n); T(:,ref)=T(:,ref)-1;
S=T*diag(1:n)*T'; A1=.2*T; F=.3*T;
e=[-2,-1,0,1,2,3,4]; d=.5*e;
C2=F*S*F'; RE=(eye(n)-A1)*S*(eye(n)-A1)';
[w,~]=build_kf2_acceptance_weights(d,C2,e,RE,ref,cfg.kf2_acceptance);
verifyEqual(testCase,w.W2,min([w.WD^(3/7)*w.WU^(3/14)*w.WA^(5/14),w.WA,.8,w.aM]),'AbsTol',1e-14);
M=pinv(RE); verifyLessThanOrEqual(testCase,(e-w.W2*d)*M*(e-w.W2*d)',e*M*e'+1e-10);
[p,CD,cross]=build_joint_deformation_uncertainty(A1,F,S,.4);
expected=A1*S*A1'+.4^2*F*S*F'+.4*(cross+cross');
verifyEqual(testCase,CD,expected,'AbsTol',1e-12);
verifyGreaterThan(testCase,norm(cross,'fro'),.1);
verifyEqual(testCase,p,diag(CD)','AbsTol',1e-12);
[zero,~]=build_kf2_acceptance_weights(zeros(1,n),C2,e,RE,ref,cfg.kf2_acceptance);
verifyEqual(testCase,zero.W2,0,'AbsTol',0);
end
function testNNLSKKTAndTruncationCounterexample(testCase)
A=[1,1;1,0]; b=[.1;3]; lambda=.10;
[x,d]=nonnegative_ridge_solve(A,b,ones(2,1),lambda);
truncated=max((A'*A+lambda*eye(2))\(A'*b),0);
obj=@(v)sum((A*v-b).^2)+lambda*sum(v.^2);
verifyLessThan(testCase,obj(x),obj(truncated)-1e-3);
verifyLessThan(testCase,d.dual_violation,1e-10);
verifyLessThan(testCase,d.complementarity,1e-10);
verifyGreaterThanOrEqual(testCase,min(x),0);
% Independent exhaustive active-set solution for small weighted subproblems.
rng(702);
for k=1:20
    A=rand(8,4); b=rand(8,1); w=rand(8,1); [x,d]=nonnegative_ridge_solve(A,b,w,.1);
    H=A'*(w.*A)+.1*eye(4); f=A'*(w.*b); best=Inf; oracle=[];
    for mask=0:15
        active=logical(bitget(mask,1:4)); z=zeros(4,1); z(active)=H(active,active)\f(active);
        if any(z<0), continue; end
        val=sum(w.*(A*z-b).^2)+.1*sum(z.^2);
        if val<best, best=val; oracle=z; end
    end
    verifyEqual(testCase,x,oracle,'AbsTol',1e-10); verifyLessThan(testCase,d.complementarity,1e-10);
end
end
function testSpatialPhysicalErrorAndCorrelatedFusion(testCase)
cfg=star_kf_v4_config(); source=[0,0;1,0;0,1]; target=[.3,.4;1,1]; r=[1;2;-.2];
p=struct('c_short',2,'c_long',.6,'rx',3,'ry',2,'rlong',6,'white',.25);
[m,v,U,K]=local_spatial_prediction(source,target,r,p,cfg.kernel,.3);
cross=matern_covariance(source,target,p,cfg.kernel);
verifyEqual(testCase,m,U'*r,'AbsTol',1e-12);
verifyEqual(testCase,v,p.c_short+p.c_long-2*sum(U.*cross,1)'+sum(U.*(K*U),1)','AbsTol',1e-12);
verifyGreaterThan(testCase,norm(v-(p.c_short+p.c_long-sum(U.*cross,1)')),1e-4);
verifyEqual(testCase,diag(K),(p.c_short+p.c_long+p.white)*ones(3,1),'AbsTol',1e-12);
alpha=[.3;.7]; errorCov=[2,1.5;1.5,3];
verifyLessThanOrEqual(testCase,alpha'*errorCov*alpha,alpha'*diag(errorCov));
verifyGreaterThan(testCase,alpha'*errorCov*alpha,(alpha.^2)'*diag(errorCov));
end
function testFeedbackMappingAndNetworkClosure(testCase)
cfg=star_kf_v4_config(); [data,~]=make_small_synthetic_case('mixed',71); cfg.i_master=data.i_master;
v=linspace(0,.3,data.n_ifg);
[R,map]=ifg_variance_to_epoch_variance(v,data.ifg_pairs,data.n_slc,data.i_master,cfg.variance_mapping);
G=map.mapper_phase;
verifyEqual(testCase,R,diag(G*diag(v)*G')','AbsTol',1e-12);
[zero,~]=ifg_variance_to_epoch_variance(zeros(size(v)),data.ifg_pairs,data.n_slc,data.i_master,cfg.variance_mapping);
verifyEqual(testCase,zero,zeros(size(zero)),'AbsTol',0);
a0=zeros(data.n_pts,data.n_slc); candidate=.01*sin(reshape(1:numel(data.phi_mat),size(data.phi_mat)));
[tau,~,a]=controlled_turbulence_update(a0,candidate,ones(size(candidate)),cfg.turbulence_update,data,G,true(size(candidate)));
[B,~,~,~]=network_incidence(data.ifg_pairs,data.n_slc,data.i_master);
verifyEqual(testCase,tau,a*B','AbsTol',1e-14);
verifyEqual(testCase,a(:,data.i_master),zeros(data.n_pts,1),'AbsTol',0);
end
function testJointStopCounterexampleSupportAndCount(testCase)
cfg=star_kf_v4_config(); data=struct('scale_rad2mm',1);
previous=struct('D_joint_mm',ones(2,3),'tau_rad',ones(2,4),'R_next_mm2',5*ones(1,3),'E_final_rad',ones(2,4));
o=compute_full_joint_convergence(2*previous.D_joint_mm,previous.tau_rad,previous.E_final_rad,previous.R_next_mm2,previous,data,cfg.convergence,4,1,true);
verifyEqual(testCase,o.Delta_E,0,'AbsTol',0); verifyFalse(testCase,o.terminate); verifyEqual(testCase,o.C,.5,'AbsTol',0);
o=compute_full_joint_convergence(previous.D_joint_mm,previous.tau_rad,previous.E_final_rad,previous.R_next_mm2,previous,data,cfg.convergence,2,0,true);
verifyEqual(testCase,o.streak,0);
o=compute_full_joint_convergence(previous.D_joint_mm,previous.tau_rad,previous.E_final_rad,previous.R_next_mm2,previous,data,cfg.convergence,3,0,true);
verifyEqual(testCase,o.streak,1); verifyFalse(testCase,o.terminate);
o=compute_full_joint_convergence(previous.D_joint_mm,previous.tau_rad,previous.E_final_rad,previous.R_next_mm2,previous,data,cfg.convergence,4,o.streak,true);
verifyTrue(testCase,o.terminate);
D=previous.D_joint_mm; D(1)=NaN;
o=compute_full_joint_convergence(D,previous.tau_rad,previous.E_final_rad,previous.R_next_mm2,previous,data,cfg.convergence,5,2,true);
verifyTrue(testCase,o.support_lost); verifyFalse(testCase,o.terminate);
verifyEqual(testCase,frobenius_relative_change(zeros(2),zeros(2)),0,'AbsTol',0);
end
function testVCEIterationCountFloorsAndRawReference(testCase)
cfg=star_kf_v4_config(); pairs=[1,2;2,3;1,3;3,4;1,4];
c=struct('c_short_ifg',[1,2,3,4,5],'c_long_ifg',[2,1,3,2,4], ...
    'sigma2_ifg',1e-4*ones(1,5),'range_x_ifg',4*ones(1,5), ...
    'range_y_ifg',5*ones(1,5),'range_long_ifg',8*ones(1,5),'known_noise_ifg_rad2',zeros(1,5));
v=vce_estimation(c,pairs,4,cfg.vce,3);
[B,~,~,T]=network_incidence(pairs,4,3);
for j=1:3
    verifyEqual(testCase,numel(v.solver_history{j}),11);
    for k=1:11
        h=v.solver_history{j}{k};
        verifyLessThan(testCase,h.primal_violation,1e-10);
        verifyLessThan(testCase,h.dual_violation,1e-9);
        verifyLessThan(testCase,h.complementarity,1e-9);
    end
end
verifyLessThan(testCase,max(v.unfloored_components(:,3)),cfg.vce.min_variance);
verifyEqual(testCase,v.sigma2_slc,.03*ones(1,4),'AbsTol',0);
verifyGreaterThan(testCase,v.c_short_slc(3),0);
verifyEqual(testCase,v.Sigma_components_relative_rad2(:,:,1),T*diag(v.c_short_slc)*T','AbsTol',1e-12);
verifyEqual(testCase,B*v.Sigma_components_relative_rad2(:,:,1)*B',B*diag(v.c_short_slc)*B','AbsTol',1e-12);
verifyEqual(testCase,rank(abs(B)),4);
[bipartite,~,~,~]=network_incidence([1,2;2,3;3,4;4,1],4,3);
verifyEqual(testCase,rank(abs(bipartite)),3);
end
function testFixedMomentSampleKnownNoiseAndNoSupport(testCase)
cfg=star_kf_v4_config(); [x,y]=meshgrid(0:7); coord=[x(:),y(:)];
r=sin((1:64)'/4)+cos((1:64)'/11);
cfg.kernel.min_valid_pts=12; cfg.kernel.shared_sample_size=64; cfg.kernel.parallel_enable=false;
[c0,cache]=stomac_covariance(coord,r,cfg.kernel,[],true(64,1),0);
[ck,again]=stomac_covariance(coord,r,cfg.kernel,cache,true(64,1),.2);
verifyEqual(testCase,cache.sample_idx,again.sample_idx,'AbsTol',0);
verifyEqual(testCase,ck.sigma2_ifg,max(c0.sigma2_ifg-.2,1e-4),'AbsTol',1e-12);
centered=r-median(r); g=centered(cache.pair_i).*centered(cache.pair_j);
weight=abs(g)./max(cache.pair_d,1.5);
expected=min(max(sum(cache.pair_dx.*weight)/sum(weight),1),35);
verifyEqual(testCase,ck.range_x_ifg,expected,'AbsTol',1e-12);
verifyError(testCase,@()stomac_covariance(coord,ones(64,1),cfg.kernel,cache,true(64,1),0),'STAR_KF:SpatialMomentSupport');
% An unsupported spatial prediction stays masked, with no zero-error fill.
v=struct('c_short_slc',[1,1],'c_long_slc',[.5,.5],'sigma2_slc',[.1,.1], ...
    'range_x_slc',[4,4],'range_y_slc',[4,4],'range_long_slc',[8,8],'known_noise_ifg_rad2',0);
kr=kriging_interpolation(coord,r,ck,v,[1,2],cfg.kriging,cfg.kernel,[],false(64,1),[]);
verifyTrue(testCase,all(isnan(kr.phi_turb))); verifyTrue(testCase,all(isnan(kr.phi_turb_var)));
end
function testExplicitInitialInputsAndInvalidVarianceInterval(testCase)
[data,~]=make_small_synthetic_case('mixed',98); cfg=small_test_config();
a0=.01*sin((1:data.n_pts)'*(1:data.n_slc)); a0(:,data.i_master)=0;
cfg.initial.atmosphere_epoch_rad=a0; cfg.initial.R_epoch_mm2=linspace(7,20,data.n_slc);
result=joint_inversion(data,cfg); first=result.iterations{1};
[B,~,~,~]=network_incidence(data.ifg_pairs,data.n_slc,data.i_master);
verifyEqual(testCase,first.R_current,cfg.initial.R_epoch_mm2,'AbsTol',0);
verifyEqual(testCase,first.observation_diagnostics.corrected_ifg_rad,data.phi_mat-a0*B','AbsTol',1e-12);
verifyGreaterThan(testCase,first.R_current(data.i_master),0);
v=struct('sigma2_slc',ones(1,data.n_slc),'known_noise_ifg_rad2',zeros(1,data.n_ifg));
verifyError(testCase,@()propagate_joint_uncertainty(ones(size(data.phi_mat)),v,ones(size(data.phi_mat)),ones(size(data.z_d_raw)),1000*ones(1,data.n_slc),data,cfg,true(size(data.phi_mat))),'STAR_KF:VarianceBounds');
end
