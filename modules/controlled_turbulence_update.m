function [tau,diagnostics,a] = controlled_turbulence_update(a_prev,candidate,var,cfg,data,Gsp,support)
% Eq. (25a,c,d): clipped IFG increments recovered to an epoch field.
[B,~,~,~]=network_incidence(data.ifg_pairs,data.n_slc,data.i_master);
tau_prev=a_prev*B'; delta=candidate-tau_prev; used=zeros(size(delta));
limits=nan(1,data.n_ifg); centers=limits; clipped=false(size(delta));
for k=1:data.n_ifg
    mask=support(:,k);
    if ~any(mask) || any(~isfinite(delta(mask,k)))
        error('STAR_KF:AtmosphereSupport','Fixed atmosphere support lost in IFG %d.',k);
    end
    x=delta(mask,k); center=median(x);
    scale=max([1.4826*median(abs(x-center)),iqr(x)/1.349,cfg.min_change_clip_rad/cfg.change_clip_sigma]);
    limit=max(cfg.change_clip_sigma*scale,cfg.min_change_clip_rad);
    used(mask,k)=center+min(max(x-center,-limit),limit);
    clipped(mask,k)=abs(x-center)>limit; limits(k)=limit; centers(k)=center;
end
valid_pixel=all(support,2); increment=zeros(size(a_prev));
increment(valid_pixel,:)=(Gsp*used(valid_pixel,:)')';
a=a_prev+cfg.beta*increment; a(:,data.i_master)=0;
tau=a*B';
diagnostics=struct('semantics','epoch update a_prev+beta*Gsp*C(delta); tau=B*a', ...
    'beta_q',cfg.beta*ones(1,data.n_ifg),'clip_limits_rad',limits,'clip_centers_rad',centers, ...
    'candidate_change_rms_rad',sqrt(mean(delta(support).^2)), ...
    'accepted_change_rms_rad',sqrt(mean((tau(support)-tau_prev(support)).^2)), ...
    'clipped_fraction',nnz(clipped)/nnz(support),'support_mask',support, ...
    'network_closure_error',norm(tau-a*B','fro'), ...
    'candidate_variance_median_rad2',median(var(support)), 'Gsp',Gsp);
end
