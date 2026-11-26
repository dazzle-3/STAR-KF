function [out,cache] = kriging_interpolation(coord,residual,covariance,vce,pairs,kcfg,kernel_cfg,point_noise,source_mask,cache)
% Eqs. (19)-(21). Direct common-source prediction and linear variance fusion.
if nargin<9 || isempty(source_mask), source_mask=true(size(coord,1),1); end
if nargin<10, cache=[]; end
if ~ismember(kernel_cfg.model_variant,{'dual_matern_anisotropic_stationary', ...
        'single_matern_anisotropic_stationary'}) || ...
        ~strcmp(kcfg.prediction_mode,'direct')
    error('STAR_KF:FormalSpatial','Formal path requires the matched Matern model and direct prediction.');
end
start=tic; np=size(coord,1); ni=size(residual,2);
settings=[kcfg.block_size_km,kcfg.overlap_ratio,kcfg.taper_ratio, ...
    kcfg.taper_min_weight,kcfg.search_radius_km,kcfg.max_neighbors,kcfg.min_sources];
reused=isstruct(cache)&&isfield(cache,'coord')&&isequaln(cache.coord,coord)&&isequaln(cache.settings,settings);
if ~reused, cache=geometry(coord,kcfg,settings); end
mu=zeros(np,ni); variance=mu; total_weight=mu; solved=0; neighbors=0;
cs=vce.c_short_slc(pairs(:,1))+vce.c_short_slc(pairs(:,2));
cl=vce.c_long_slc(pairs(:,1))+vce.c_long_slc(pairs(:,2));
white=vce.sigma2_slc(pairs(:,1))+vce.sigma2_slc(pairs(:,2));
rx=mean(vce.range_x_slc(pairs),2); ry=mean(vce.range_y_slc(pairs),2);
rl=mean(vce.range_long_slc(pairs),2);
loading_used=nan(numel(cache.blocks),ni);
for b=1:numel(cache.blocks)
    target=cache.blocks{b}; pool=cache.sources{b}; taper=cache.taper{b};
    for k=1:ni
        source=pool(source_mask(pool)&isfinite(residual(pool,k)));
        source=source(1:min(numel(source),kcfg.max_neighbors));
        if numel(source)<kcfg.min_sources, continue; end
        p=struct('c_short',cs(k),'c_long',cl(k),'rx',rx(k),'ry',ry(k), ...
            'rlong',rl(k),'white',white(k)+vce.known_noise_ifg_rad2(k));
        % Exact existing no-enhancement loaded matrix, with physical noise separated.
        loading=max(white(k)+kcfg.nugget,kcfg.solver_reg_fraction*max(cs(k)+cl(k),kcfg.nugget))-white(k);
        if loading<=0, error('STAR_KF:SpatialLoading','Positive spatial loading required.'); end
        [pred,var]=local_spatial_prediction(coord(source,:),coord(target,:),residual(source,k),p,kernel_cfg,loading);
        valid=isfinite(pred)&isfinite(var); ids=target(valid); w=taper(valid);
        mu(ids,k)=mu(ids,k)+w.*pred(valid); variance(ids,k)=variance(ids,k)+w.*var(valid);
        total_weight(ids,k)=total_weight(ids,k)+w;
        loading_used(b,k)=loading; solved=solved+1; neighbors=neighbors+numel(source);
    end
end
support=total_weight>0;
mu(support)=mu(support)./total_weight(support); variance(support)=variance(support)./total_weight(support);
mu(~support)=NaN; variance(~support)=NaN;
out=struct('phi_turb',mu,'phi_turb_var',variance,'support_mask',support, ...
    'loading_rad2',loading_used,'quality',struct('mean_neighbors',neighbors/max(solved,1), ...
        'n_blocks',numel(cache.blocks),'solved_count',solved,'fallback_count',0,'clipped_count',0, ...
        'geometry_cache_reused',reused,'prediction_mode','direct','elapsed_sec',toc(start), ...
        'variance_semantics','linear convex fusion of physical latent prediction errors', ...
        'unsupported_count',nnz(~support)));
log_progress('KRIG','Direct prediction: blocks=%d, supported=%.4f, elapsed=%.2f s',numel(cache.blocks),mean(support(:)),toc(start));
end
function cache=geometry(coord,cfg,settings)
step=cfg.block_size_km*(1-cfg.overlap_ratio); half=cfg.block_size_km/2;
xs=min(coord(:,1)):step:max(coord(:,1))+step;
ys=min(coord(:,2)):step:max(coord(:,2))+step;
blocks={}; boxes={}; sources={}; taper={};
for x=xs
    for y=ys
        box=[x-half,x+half,y-half,y+half];
        idx=find(coord(:,1)>=box(1)&coord(:,1)<=box(2)&coord(:,2)>=box(3)&coord(:,2)<=box(4));
        if isempty(idx), continue; end
        distance=hypot(coord(:,1)-x,coord(:,2)-y);
        pool=find(distance<=cfg.search_radius_km);
        ordered=sortrows([distance(pool),pool],[1,2]); pool=ordered(:,2);
        d=min([coord(idx,1)-box(1),box(2)-coord(idx,1),coord(idx,2)-box(3),box(4)-coord(idx,2)],[],2);
        z=min(d/(cfg.block_size_km*cfg.taper_ratio),1);
        w=cfg.taper_min_weight+(1-cfg.taper_min_weight)*(1-cos(pi*z))/2;
        blocks{end+1}=idx; boxes{end+1}=box; sources{end+1}=pool; taper{end+1}=w; %#ok<AGROW>
    end
end
cache=struct('coord',coord,'settings',settings,'blocks',{blocks},'boxes',{boxes},'sources',{sources},'taper',{taper});
end
