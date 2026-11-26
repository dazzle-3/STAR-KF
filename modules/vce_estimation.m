function out = vce_estimation(covariance,pairs,n,params,ref)
if nargin<5, ref=1; end
[B,~,~,T]=network_incidence(pairs,n,ref); A=abs(B);
fields={'c_short_ifg','c_long_ifg','sigma2_ifg'};
raw=zeros(n,3); history=cell(1,3);
single=isfield(covariance,'quality') && isfield(covariance.quality,'model_variant') ...
    && strcmp(covariance.quality.model_variant,'single_matern_anisotropic_stationary');
active=1:3;
if single
    assert(all(covariance.c_long_ifg==0),'Single Matern must have structural-zero long moments.');
    active=[1,3]; % Refit two independent robust VCE problems; no long solve.
end
for j=active
    b=covariance.(fields{j})(:); valid=isfinite(b);
    if ~any(valid), error('STAR_KF:VCESupport','No valid VCE observations.'); end
    Av=A(valid,:); b=b(valid); w=ones(size(b));
    [x,h]=nonnegative_ridge_solve(Av,b,w,params.lambda_reg);
    hist=cell(params.robust_iters+1,1); hist{1}=h;
    for it=1:params.robust_iters
        resid=Av*x-b; scale=median(abs(resid))/.6745;
        if ~isfinite(scale)||scale<=0, scale=1; end % documented existing zero-scale convention
        u=resid/scale;
        student=(params.student_df+1)./(params.student_df+u.^2);
        huber=min(1,params.huber_k./abs(u));
        w=max(student.*huber,1e-3);
        [x,h]=nonnegative_ridge_solve(Av,b,w,params.lambda_reg);
        h.robust_scale=scale; hist{it+1}=h;
    end
    raw(:,j)=x; history{j}=hist;
end
used=raw; used(:,3)=max(raw(:,3),params.min_variance);
rel=zeros(n,n,3); ifgcov=zeros(size(B,1),size(B,1),3);
for j=1:3
    rel(:,:,j)=T*diag(used(:,j))*T'; ifgcov(:,:,j)=B*diag(used(:,j))*B';
end
out=struct('c_short_slc',used(:,1)','c_long_slc',used(:,2)', ...
    'sigma2_slc',used(:,3)','unfloored_components',raw,'solver_history',{history}, ...
    'Sigma_components_relative_rad2',rel,'Sigma_components_ifg_rad2',ifgcov, ...
    'range_x_slc',ranges(covariance.range_x_ifg,pairs,n,params.d0_smooth), ...
    'range_y_slc',ranges(covariance.range_y_ifg,pairs,n,params.d0_smooth), ...
    'range_long_slc',ranges(covariance.range_long_ifg,pairs,n,params.d0_smooth), ...
    'known_noise_ifg_rad2',covariance.known_noise_ifg_rad2, ...
    'quality',struct('mean_sigma2_slc',mean(used(:,3)), ...
        'unsigned_incidence_rank',rank(A),'data_identifiable',rank(A)==n, ...
        'subproblem_unique',params.lambda_reg>0,'reweighted_global_optimality_claimed',false, ...
        'active_component_indices',active,'long_component_estimated',~single));
end
function out=ranges(v,pairs,n,window)
total=zeros(n,1); count=total;
for k=1:numel(v)
    if ~isfinite(v(k)), continue; end
    ids=pairs(k,:); total(ids)=total(ids)+v(k); count(ids)=count(ids)+1;
end
if any(count==0), error('STAR_KF:RangeSupport','Epoch range lacks support.'); end
out=total./count;
if window>1 && n>window, out=conv(out,ones(window,1)/window,'same'); end
out=out';
end
