function enabled = ablation_enabled(cfg, name)
% Backward compatibility for saved full configurations without ablation fields.
enabled = true;
if isfield(cfg,'ablation') && isfield(cfg.ablation,name)
    enabled = cfg.ablation.(name);
end
if ~islogical(enabled) || ~isscalar(enabled)
    error('STAR_KF:AblationControl','cfg.ablation.%s must be a logical scalar.',name);
end
end
