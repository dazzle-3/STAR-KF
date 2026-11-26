function [prediction,variance,U,K] = local_spatial_prediction(source,target,r,p,kcfg,loading)
% Eq. (19)-(20): loading changes U only; noise enters physical K once.
C=matern_covariance(source,source,p,kcfg);
K=C+(p.white)*eye(size(C)); cross=matern_covariance(source,target,p,kcfg);
L=chol((K+K')/2+loading*eye(size(K)),'lower'); U=L'\(L\cross);
prediction=(U'*r(:));
variance=p.c_short+p.c_long-2*sum(U.*cross,1)'+sum(U.*(K*U),1)';
if any(variance < -1e-9*max(1,p.c_short+p.c_long))
    error('STAR_KF:PredictionVariance','Negative physical prediction error.');
end
variance=max(variance,0); % roundoff only
end
