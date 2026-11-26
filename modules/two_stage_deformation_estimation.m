function result = two_stage_deformation_estimation(Y,data,R_current,cfg,network)
% Existing two-stage wrapper; process each fixed-fit covariance in bounded memory.
if nargin<5 || ~isfield(network,'SigmaY')
    error('STAR_KF:FullCovariance','Two-stage estimation requires propagated full SigmaY.');
end
tic_stage=tic; [np,n]=size(Y); ref=cfg.i_master;
t=data.acq_times(:)'-data.acq_times(1); names={'cv','ca','seasonal'};
cache1=cell(1,3); Kd=cell(1,3);
for m=1:3
    cache1{m}=build_kalman_model_cache(t,cfg.kalman,names{m});
    if ablation_enabled(cfg,'kf2_enable')
        c2=build_kalman_model_cache(t,cfg.kalman_kf2,names{m});
        Kd{m}=build_trajectory_prior(c2,cfg.kalman_kf2.q_acc,ref);
    end
end
parallel=ensure_star_kf_parallel(cfg.kalman,np);
values=zeros(np,n,8); weight_values=zeros(np,5);
alpha1=zeros(np,n,3); alpha2=alpha1;
retain=false(np,1);
if cfg.operator_output.retain_all, retain(:)=true; else
    idx=cfg.operator_output.sample_pixel_indices;
    retain(idx(idx>=1 & idx<=np))=true;
end
saved=cell(np,1); whitening_error=zeros(np,1); operator_error=zeros(np,1);
chunk=max(1,min(cfg.kalman.parallel_chunk_size,2048));
for first=1:chunk:np
    ids=first:min(first+chunk-1,np); fits=cell(numel(ids),1);
    Yblock=Y(ids,:); patterns=network.pattern_index(ids); keeps=retain(ids);
    Sigma=network.SigmaY;
    if parallel
        parfor j=1:numel(ids)
            fits{j}=estimate_pixel(Yblock(j,:),Sigma(:,:,patterns(j)),t, ...
                cfg,ref,cache1,Kd,keeps(j));
        end
    else
        for j=1:numel(ids)
            fits{j}=estimate_pixel(Yblock(j,:),Sigma(:,:,patterns(j)),t, ...
                cfg,ref,cache1,Kd,keeps(j));
        end
    end
    for j=1:numel(ids)
        p=ids(j); f=fits{j}; values(p,:,:)=reshape(f.values,[1,n,8]);
        weight_values(p,:)=f.weights; alpha1(p,:,:)=reshape(f.alpha1',[1,n,3]);
        alpha2(p,:,:)=reshape(f.alpha2',[1,n,3]); saved{p}=f.operators;
        whitening_error(p)=f.whitening_error; operator_error(p)=f.operator_error;
    end
end
weights=struct('WD',weight_values(:,1),'WU',weight_values(:,2), ...
    'WA',weight_values(:,3),'aM',weight_values(:,4),'W2',weight_values(:,5));
runtime=struct('elapsed_sec',toc(tic_stage),'parallel_used',parallel,'chunk_size',chunk);
kf1=filter_summary(values(:,:,1),values(:,:,2),alpha1,runtime);
kf2=filter_summary(values(:,:,4),values(:,:,5),alpha2,runtime);
candidate=mean(values(:,:,4).^2,'all'); accepted=mean(values(:,:,6).^2,'all');
result=struct('D1_q',values(:,:,1),'P1_q',values(:,:,2),'E1_q',values(:,:,3), ...
    'D2_raw_q',values(:,:,4),'P2_q',values(:,:,5),'D2_q',values(:,:,6), ...
    'D_joint_q',values(:,:,7),'P_joint_q',values(:,:,8),'weights',weights, ...
    'kf1',kf1,'kf2',kf2,'operator_pixel_indices',find(retain), ...
    'operators',{saved(retain)}, ...
    'diagnostics',struct('candidate_energy_mm2',candidate,'accepted_energy_mm2',accepted, ...
        'accepted_to_candidate_energy_ratio',accepted/max(candidate,eps), ...
        'acceptance',struct('accepted_pixel_fraction',mean(weights.W2>=cfg.kf2_acceptance.diagnostic_acceptance_weight)), ...
        'max_whitening_error',max(whitening_error),'max_A1_reconstruction_error',max(operator_error), ...
        'covariance_semantics',cfg.output.deformation_covariance_semantics, ...
        'kf2_enabled',ablation_enabled(cfg,'kf2_enable'), ...
        'acceptance_enabled',ablation_enabled(cfg,'acceptance_enable')));
end
function out=estimate_pixel(y,S,t,cfg,ref,cache1,Kd,retain)
f1=fit_kf1_pixel(y,S,t,cfg.kalman,ref,cache1); e=y-f1.sequence;
if ablation_enabled(cfg,'kf2_enable')
    f2=correlated_residual_filter(e,f1.A,S,Kd,cfg.kalman_kf2,ref);
    if ablation_enabled(cfg,'acceptance_enable')
        [w,~]=build_kf2_acceptance_weights(f2.sequence,f2.C2,e,f2.RE,ref,cfg.kf2_acceptance);
    else
        % Unit acceptance bypasses all support scores and projection caps.
        % KF2 internal robust observation updates above remain untouched.
        finite_candidate=all(isfinite(f2.sequence)) && all(isfinite(f2.C2(:))) ...
            && all(isfinite(f2.F2(:))) && isfinite(f2.whitening_error) ...
            && any(f2.variance(setdiff(1:numel(y),ref))>0);
        u=double(finite_candidate);
        w=struct('WD',u,'WU',u,'WA',u,'aM',u,'W2',u);
        if ~finite_candidate
            % An invalid candidate must not propagate through 0*NaN.
            f2.sequence=zeros(size(y)); f2.variance=zeros(size(y));
            f2.F2=zeros(size(S)); f2.C2=zeros(size(S));
        end
    end
else
    % Structural zeros are placeholders only: no KF2 estimation is executed.
    f2=struct('sequence',zeros(size(y)),'variance',zeros(size(y)), ...
        'F2',zeros(size(S)),'C2',zeros(size(S)),'alpha',zeros(size(f1.alpha)), ...
        'whitening_error',0,'enabled',false);
    w=struct('WD',0,'WU',0,'WA',0,'aM',0,'W2',0);
end
[P,CD,cross]=build_joint_deformation_uncertainty(f1.A,f2.F2,S,w.W2);
accepted=w.W2*f2.sequence; joint=f1.sequence+accepted;
operators=[];
if retain
    operators=struct('SigmaY',S,'A1',f1.A,'KF1',{f1.models},'KF2',f2, ...
        'C1',f1.A*S*f1.A','CD',CD,'cross_C1_C2',cross,'w',w.W2);
end
seqs=cellfun(@(r)r.sequence_mm(:),f1.models,'UniformOutput',false);
independent=sum(cell2mat(seqs).*f1.alpha',2)';
out=struct('values',[f1.sequence;f1.variance;e;f2.sequence;f2.variance;accepted;joint;P]', ...
    'weights',[w.WD,w.WU,w.WA,w.aM,w.W2],'alpha1',f1.alpha,'alpha2',f2.alpha, ...
    'operators',operators,'whitening_error',f2.whitening_error, ...
    'operator_error',norm(independent-f1.sequence));
end
