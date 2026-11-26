function out = fit_kf1_pixel(y,SigmaY,t,params,ref,caches)
n=numel(y); nm=numel(caches); scores=zeros(nm,n); Astack=zeros(n,n,nm);
models=cell(1,nm); residual=zeros(nm,n); posterior=zeros(nm,n);
for m=1:nm
    r=kalman_filter(y,t,diag(SigmaY)',params,caches{m}.model,ref,caches{m});
    scores(m,:)=r.step_loglike; Astack(:,:,m)=r.A;
    models{m}=r; residual(m,:)=r.residual_mm; posterior(m,:)=r.state_var_mm2;
end
[alpha,mu,xi]=predictive_model_weights(scores,params);
A=zeros(n);
for m=1:nm, A=A+alpha(m,:)'.*Astack(:,:,m); end
ysafe=y(:); ysafe(~isfinite(ysafe))=0;
C=A*SigmaY*A';
out=struct('A',A,'sequence',(A*ysafe)','variance',max(diag(C)',0), ...
    'alpha',alpha,'mu',mu,'xi',xi,'models',{models},'scores',scores, ...
    'posterior_diagnostic',posterior,'residual',sum(alpha.*residual,1));
end
