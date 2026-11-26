function [seq,operators] = recover_sequence_phase_weighted(phi,n,pairs,ref,noise,lambda)
% Missing rows use inverse marginal covariance, never a full-precision block.
% Covariance outputs require an explicit physical model, separating loading.
if nargout>1 && ~isstruct(noise)
    error('STAR_KF:PhysicalCovarianceRequired', ...
        'Covariance outputs require a physical covariance model with explicit loading.');
end
if nargin<6, lambda=0; end
m=size(pairs,1); [~,J,Br]=network_incidence(pairs,n,ref);
physical=[]; loading=0;
if isstruct(noise)
    physical=noise.Dyy; loading=noise.eps_reg;
    solve_cov=physical+loading*eye(m);
elseif isempty(noise)
    solve_cov=eye(m);
elseif isvector(noise)
    solve_cov=diag(noise(:));
else
    solve_cov=noise\eye(m);
end
[masks,~,pattern]=unique(isfinite(phi),'rows');
seq=nan(size(phi,1),n);
Gphase=zeros(n,m,size(masks,1)); SigmaPhase=zeros(n,n,size(masks,1));
for j=1:size(masks,1)
    valid=masks(j,:); Bv=Br(valid,:);
    if rank(Bv)<n-1
        error('STAR_KF:NetworkSupport','Observed IFG network disconnected (pattern %d).',j);
    end
    Cv=solve_cov(valid,valid); C=chol((Cv+Cv')/2,'lower');
    P=C'\(C\eye(nnz(valid)));
    mapper=J*((Bv'*P*Bv+lambda*eye(n-1))\(Bv'*P));
    Gphase(:,valid,j)=mapper; rows=find(pattern==j);
    seq(rows,:)=(mapper*phi(rows,valid)')';
    if ~isempty(physical)
        S=mapper*physical(valid,valid)*mapper'; SigmaPhase(:,:,j)=(S+S')/2;
    end
end
operators=struct('Gphase',Gphase,'SigmaPhase',SigmaPhase, ...
    'pattern_index',pattern,'valid_ifg_patterns',masks,'loading_rad2',loading);
end
