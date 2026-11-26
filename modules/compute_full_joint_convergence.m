function out = compute_full_joint_convergence(D,tau,E,R,previous,data,cfg,q,streak,fresh)
% Eqs. (26)-(28). Residual-energy change is diagnostic only.
if nargin<10, fresh=true; end
lost=false;
if isempty(previous)
    dD=NaN; dt=NaN; dR=NaN; dE=NaN; C=Inf;
else
    if isfield(previous,'support')
        mask=previous.support;
    else
        mask=struct('D',isfinite(previous.D_joint_mm),'tau',isfinite(previous.tau_rad),'R',isfinite(previous.R_next_mm2));
    end
    [dD,l1]=frobenius_relative_change(D,previous.D_joint_mm,mask.D);
    [dt,l2]=frobenius_relative_change(tau,previous.tau_rad,mask.tau);
    [dR,l3]=frobenius_relative_change(R,previous.R_next_mm2,mask.R);
    lost=l1||l2||l3; C=max([dD,dt,dR]);
    a=mean(E(isfinite(E)).^2); b=mean(previous.E_final_rad(isfinite(previous.E_final_rad)).^2);
    if max(a,b)==0, dE=0; else, dE=abs(a-b)/max(a,b); end
end
eligible=fresh && ~lost && q>=cfg.min_outer_iters && C<cfg.joint_state_tol;
if eligible, streak=streak+1; else, streak=0; end
out=struct('Delta_D',dD,'Delta_tau',dt,'Delta_R',dR,'Delta_E',dE, ...
    'C',C,'C_diagnostic',C,'termination_metric','max_Frobenius_D_tau_R', ...
    'thresholds',struct('C',cfg.joint_state_tol),'support_lost',lost, ...
    'fresh_spatial_update',fresh,'eligible_this_iteration',eligible, ...
    'streak',streak,'terminate',eligible && streak>=cfg.consecutive_converged_iters);
end
