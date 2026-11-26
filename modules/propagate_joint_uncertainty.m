function [R_next,components] = propagate_joint_uncertainty(vsp,vce,E,CDdiag,R,data,cfg,support)
% Eqs. (22)-(25): four overlapping working contributions, not independent errors.
if nargin<9, support=isfinite(E)&isfinite(vsp); end
sp=zeros(1,data.n_ifg); energy=sp;
for k=1:data.n_ifg
    mask=support(:,k);
    if ~any(mask) || any(~isfinite(vsp(mask,k))) || any(~isfinite(E(mask,k)))
        error('STAR_KF:FeedbackSupport','Fixed common feedback support lost.');
    end
    sp(k)=mean(vsp(mask,k)); x=E(mask,k); limit=prctile(abs(x),cfg.uncertainty.residual_clip_percentile);
    energy(k)=mean(min(x.^2,limit^2));
end
white=vce.sigma2_slc(data.ifg_pairs(:,1))+vce.sigma2_slc(data.ifg_pairs(:,2)); white=white(:)';
white=white+vce.known_noise_ifg_rad2;
res=max(energy-white-sp,0);
[Rsp,spmap]=ifg_variance_to_epoch_variance(sp,data.ifg_pairs,data.n_slc,data.i_master,cfg.variance_mapping);
[Rres,resmap]=ifg_variance_to_epoch_variance(res,data.ifg_pairs,data.n_slc,data.i_master,cfg.variance_mapping);
Rsp=Rsp*data.scale_rad2mm^2; Rres=Rres*data.scale_rad2mm^2;
Rvce=vce.sigma2_slc*data.scale_rad2mm^2;
raw_def=median(CDdiag,1,'omitnan'); base=Rvce+Rsp+Rres; u=cfg.uncertainty;
Rdef=min(u.deformation_participation*raw_def,u.deformation_cap_fraction*max(base,u.r_floor_mm2));
target=base+Rdef;
feedback_enabled=ablation_enabled(cfg,'uncertainty_feedback_enable');
if ~feedback_enabled
    target=Rvce; % No spatial, residual or deformation term enters feedback.
end
low=max(u.r_floor_mm2,u.min_round_ratio*R);
high=min(u.r_ceiling_mm2,u.max_round_ratio*max(R,u.r_floor_mm2));
if any(low>high), error('STAR_KF:VarianceBounds','Given R makes Eq. (25f) interval empty.'); end
bounded=min(max(target,low),high);
R_next=(1-u.round_damping)*R+u.round_damping*bounded;
components=struct('slc_vce_mm2',Rvce,'slc_kriging_latent_mm2',Rsp, ...
    'slc_residual_positive_mm2',Rres,'slc_deformation_participation_mm2',Rdef, ...
    'slc_joint_raw_mm2',raw_def,'raw_target_mm2',target,'bounded_target_mm2',bounded, ...
    'ifg_kriging_total_rad2',sp,'ifg_kriging_latent_rad2',sp, ...
    'ifg_vce_white_rad2',white,'ifg_residual_energy_rad2',energy,'ifg_residual_positive_rad2',res, ...
    'kriging_variance_includes_vce_and_nugget',false,'kriging_network_mapping',spmap, ...
    'residual_network_mapping',resmap,'round_damping',u.round_damping, ...
    'uncertainty_feedback_enabled',feedback_enabled, ...
    'semantics','overlapping fixed-fit working contributions; omitted cross-component/IFG/iteration covariances', ...
    'deformation_covariance_semantics',cfg.output.deformation_covariance_semantics);
end
