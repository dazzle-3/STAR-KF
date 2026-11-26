function result = multi_model_filter(Y,t,params,observation,ref)
% Separate KF1 model estimates with Eq. (6) forward/backward score weights.
[np,n]=size(Y); t=t(:)'-t(1); names={'cv','ca','seasonal'};
caches=cell(1,3);
for m=1:3, caches{m}=build_kalman_model_cache(t,params,names{m}); end
if isvector(observation), S=diag(observation); else, S=observation; end
seq=zeros(np,n); var=seq; alpha=zeros(np,n,3); operators=zeros(n,n,np);
parallel=ensure_star_kf_parallel(params,np);
if parallel
    parfor p=1:np
        fit=fit_kf1_pixel(Y(p,:),S(:,:,min(p,size(S,3))),t,params,ref,caches);
        seq(p,:)=fit.sequence; var(p,:)=fit.variance;
        alpha(p,:,:)=reshape(fit.alpha',[1,n,3]); operators(:,:,p)=fit.A;
    end
else
    for p=1:np
        fit=fit_kf1_pixel(Y(p,:),S(:,:,min(p,size(S,3))),t,params,ref,caches);
        seq(p,:)=fit.sequence; var(p,:)=fit.variance;
        alpha(p,:,:)=reshape(fit.alpha',[1,n,3]); operators(:,:,p)=fit.A;
    end
end
result=filter_summary(seq,var,alpha,struct('parallel_used',parallel)); result.A1=operators;
end
