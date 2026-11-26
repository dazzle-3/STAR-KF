function Kd = build_trajectory_prior(cache,q,ref)
% Unconditional trajectory covariance includes P0 and independent process Q.
n=cache.n_epoch; d=cache.dim; cross=zeros(d,d,n,n);
cross(:,:,1,1)=cache.P0;
for k=2:n
    F=cache.F(:,:,k);
    cross(:,:,k,k)=F*cross(:,:,k-1,k-1)*F'+q*cache.Q0(:,:,k);
    for l=1:k-1
        cross(:,:,k,l)=F*cross(:,:,k-1,l);
        cross(:,:,l,k)=cross(:,:,k,l)';
    end
end
K=zeros(n);
for k=1:n
    for l=1:n, K(k,l)=cache.H(k,:)*cross(:,:,k,l)*cache.H(l,:)'; end
end
if isempty(ref)
    Kd=K;
else
    T=eye(n); T(:,ref)=T(:,ref)-1; Kd=T*K*T';
end
Kd=(Kd+Kd')/2;
end
