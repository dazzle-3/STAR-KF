function C = matern_covariance(a,b,p,kcfg)
% Eq. (17a-b), stationary short anisotropy and isotropic long component.
dx=a(:,1)-b(:,1)'; dy=a(:,2)-b(:,2)';
rS=hypot(dx/p.rx,dy/p.ry);
if strcmp(kcfg.model_variant,'single_matern_anisotropic_stationary')
    assert(p.c_long==0,'Single Matern cannot contain a long component.');
    C=p.c_short*matern(rS,kcfg.matern_nu_short);
else
    rL=hypot(dx,dy)/p.rlong;
    C=p.c_short*matern(rS,kcfg.matern_nu_short)+p.c_long*matern(rL,kcfg.matern_nu_long);
end
end
function K=matern(r,nu)
z=sqrt(2*nu)*r; K=ones(size(r)); ix=z>0;
if abs(nu-1.5)<1e-12
    K(ix)=(1+z(ix)).*exp(-z(ix));
else
    K(ix)=2^(1-nu)/gamma(nu)*z(ix).^nu.*besselk(nu,z(ix));
end
end
