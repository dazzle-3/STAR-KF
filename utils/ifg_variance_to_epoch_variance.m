function [epoch_var,diagnostics] = ifg_variance_to_epoch_variance(v,pairs,n,ref,settings)
% Eq. (22) in phase units. Caller performs the single kappa^-2 conversion.
if nargin<5
    error('STAR_KF:MappingSettings','Explicit variance-mapping run settings required.');
end
v=double(v(:));
if numel(v)~=size(pairs,1) || any(~isfinite(v)|v<0)
    error('STAR_KF:VarianceSupport','Mapping requires a finite nonnegative IFG vector.');
end
[~,J,Br]=network_incidence(pairs,n,ref);
if rank(Br)<n-1, error('STAR_KF:NetworkSupport','Disconnected variance network.'); end
positive=v(v>settings.absolute_floor_rad2);
if isempty(positive), floor_v=settings.absolute_floor_rad2; else
    floor_v=max(median(positive)*settings.floor_fraction,settings.absolute_floor_rad2);
end
weight=1./max(v,floor_v); normal=Br'*(weight.*Br);
lambda=settings.regularization_fraction*median(diag(normal));
G=J*((normal+lambda*eye(n-1))\(Br'.*weight'));
C=(G.*v')*G'; epoch_var=max(diag(C)',0);
diagnostics=struct('rank',rank(Br),'n_valid_ifg',numel(v),'regularization',lambda, ...
    'solver_variance_floor',floor_v,'zero_variance_count',nnz(v==0), ...
    'mapper_phase',G,'covariance_phase_rad2',C,'method','unloaded_linear_mapper_covariance', ...
    'units','rad2 input and output; phase-to-phase G');
end
