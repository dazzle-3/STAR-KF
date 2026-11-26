function [alpha,mu,xi] = predictive_model_weights(scores,params)
% Eq. (6a-e). Models are estimated separately; these are score weights.
[nm,n]=size(scores);
if any(~isfinite(scores(:))), error('STAR_KF:Scores','Scores must be finite.'); end
prior=params.model_prior(:);
if numel(prior)~=nm || any(prior<=0) || abs(sum(prior)-1)>1e-12
    error('STAR_KF:ModelPrior','Prescribed model prior does not match models.');
end
if nm==1, T=1; else
    T=ones(nm)*(1-params.imm_self_prob)/(nm-1);
    T(1:nm+1:end)=params.imm_self_prob;
end
mu=zeros(nm,n); xi=zeros(nm,nm,n); prev=prior;
for k=1:n
    c=T'*prev;
    xi(:,:,k)=(T.*prev)./c';
    logw=log(c)+scores(:,k)/params.score_temperature;
    w=exp(logw-max(logw)); mu(:,k)=w/sum(w); prev=mu(:,k);
end
alpha=mu;
for k=n-1:-1:1
    alpha(:,k)=xi(:,:,k+1)*alpha(:,k+1);
    alpha(:,k)=alpha(:,k)/sum(alpha(:,k));
end
end
