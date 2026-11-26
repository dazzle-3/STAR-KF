function model = build_ifg_covariance_model(pairs,n,ref,slc_var,extra,reg_fraction)
% Eq. (3a): raw acquisition variances in rad^2, including the reference.
[B,~,~,~]=network_incidence(pairs,n,ref);
slc_var=double(slc_var(:));
if numel(slc_var)~=n || any(~isfinite(slc_var)|slc_var<0)
    error('STAR_KF:CovarianceInput','Invalid raw epoch variances.');
end
m=size(B,1);
if isscalar(extra), extra=repmat(extra,m,1); end
if isvector(extra), QI=diag(extra(:)); else, QI=extra; end
if ~isequal(size(QI),[m,m]) || any(~isfinite(QI(:))) || ...
        norm(QI-QI','fro')>1e-10 || min(eig((QI+QI')/2)) < -1e-10
    error('STAR_KF:CovarianceInput','QI must be a finite PSD phase covariance.');
end
Dyy=B*diag(slc_var)*B'+QI;
eps_reg=reg_fraction*median(diag(Dyy));
if ~isfinite(eps_reg) || eps_reg<=0
    error('STAR_KF:Loading','Positive phase covariance loading is required.');
end
Dload=Dyy+eps_reg*eye(m);
C=chol((Dload+Dload')/2,'lower'); Pyy=C'\(C\eye(m));
model=struct('B',B,'QI',QI,'Dyy',Dyy,'Dyy_regularized',Dload, ...
    'Pyy',Pyy,'diag_var',diag(Dyy),'eps_reg',eps_reg,'ref_epoch',ref);
end
