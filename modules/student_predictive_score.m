function [score,a,u] = student_predictive_score(v,S0,params)
% Eq. (6f): variance-matched common Student-t, before current inflation.
nu=params.student_df;
if nu<=2 || ~isfinite(S0) || S0<=0
    error('STAR_KF:PredictiveVariance','Student-t scoring requires nu>2 and S0>0.');
end
b2=(nu-2)*S0/nu;
score=gammaln((nu+1)/2)-gammaln(nu/2)-.5*log(nu*pi*b2) ...
    -(nu+1)/2*log1p(v^2/(nu*b2));
u=abs(v)/sqrt(S0);
a=max([1,(nu+u^2)/(nu+1),u/params.huber_k]);
end
