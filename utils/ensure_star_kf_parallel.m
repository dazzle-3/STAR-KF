function tf = ensure_star_kf_parallel(params,np)
tf=params.parallel_enable && np>512 && license('test','Distrib_Computing_Toolbox');
if ~tf, return; end
try
    if isempty(gcp('nocreate'))
        parpool(params.parallel_pool_type,params.parallel_worker_count);
    end
catch ex
    warning('STAR_KF:ParallelUnavailable','Serial fallback: %s',ex.message); tf=false;
end
end
