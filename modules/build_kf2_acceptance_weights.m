function [weights,diagnostics] = build_kf2_acceptance_weights(delta,C2,e,RE,ref,cfg)
% Eqs. (11)-(14), exactly three empirical support components.
% Arrays are pixels x epochs, covariance is epochs x epochs x pixels.
[np,n]=size(delta); WD=zeros(np,1); WU=WD; WA=WD; aM=WD; w=WD;
eta=nan(np,1); rho=eta; z=eta; frac=eta;
for p=1:np
    d=delta(p,:)'; r=e(p,:)';
    S=C2(:,:,min(p,size(C2,3))); R=RE(:,:,min(p,size(RE,3)));
    v=diag(S); valid=isfinite(d)&isfinite(r)&isfinite(v)&v>0; valid(ref)=false;
    if ~any(valid) || any(~isfinite(d)) || any(~isfinite(r)), continue; end
    M=pinv((R+R')/2); er=r'*M*r; ed=d'*M*d;
    if er<=0 || ed<=0 || ~isfinite(er+ed), continue; end
    cross=d'*M*r;
    eta(p)=1-(r-d)'*M*(r-d)/er; rho(p)=cross/sqrt(ed*er);
    WD(p)=sqrt(h(eta(p),cfg.support_gain_low,cfg.support_gain_high)* ...
        h(rho(p),cfg.support_corr_low,cfg.support_corr_high));
    nv=nnz(valid); z(p)=sqrt(mean(d(valid).^2)/mean(v(valid)));
    frac(p)=mean(abs(d(valid))>=cfg.significance_z_low*sqrt(v(valid)));
    WU(p)=.70*h(z(p)*sqrt(min(nv,cfg.significance_effective_epoch_cap)), ...
        cfg.significance_z_low,cfg.significance_z_high)+ ...
        .30*h(frac(p),cfg.significant_fraction_low,cfg.significant_fraction_high);
    dr=sqrt(mean(d.^2)); rr=sqrt(mean(r.^2));
    if dr<=0, continue; end
    range_d=max(d)-min(d); range_r=max(r)-min(r);
    cap=Inf; if range_d>0, cap=cfg.max_range_recovery_fraction*range_r/range_d; end
    WA(p)=min([1,cfg.max_rms_recovery_fraction*rr/dr,cap]);
    aM(p)=min(max(cross/ed,0),1);
    values=[WD(p),WU(p),WA(p),aM(p)];
    if all(isfinite(values))
        w(p)=min([WD(p)^(3/7)*WU(p)^(3/14)*WA(p)^(5/14), ...
            WA(p),cfg.max_pixel_weight,aM(p)]);
    end
end
weights=struct('WD',WD,'WU',WU,'WA',WA,'aM',aM,'W2',w);
diagnostics=struct('eta_M',eta,'rho_M',rho,'z',z,'significant_fraction',frac, ...
    'accepted_pixel_fraction',mean(w>=cfg.diagnostic_acceptance_weight), ...
    'mean_W2',mean(w),'median_W2',median(w),'p90_W2',prctile(w,90));
end
function y=h(x,l,u)
s=min(max((x-l)/(u-l),0),1); y=s^2*(3-2*s);
end
