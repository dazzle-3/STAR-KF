function result = extract_leakage_imm_kalman(A_final_turb_mm, acq_times, cfg, i_master)
if nargin < 3 || isempty(cfg)
    cfg = struct();
end
if nargin < 4 || isempty(i_master)
    i_master = 1;
end

imm_cfg = default_imm_config();
if isstruct(cfg) && isfield(cfg, 'imm') && isstruct(cfg.imm)
    imm_cfg = merge_struct(imm_cfg, cfg.imm);
end

imm_cfg.enabled_models = {'cv', 'ca'};
imm_cfg.model_type = 'probabilistic';
imm_cfg.fusion_mode = 'imm';
imm_cfg.do_rts = true;
imm_cfg.model_prior = normalize_model_prior(get_param(imm_cfg, 'model_prior', [0.5, 0.5]));

n_epoch = size(A_final_turb_mm, 2);
observation_var = get_param(cfg, 'observation_var_mm2', []);
if isempty(observation_var)
    observation_var = imm_cfg.R_floor * ones(1, n_epoch);
elseif isscalar(observation_var)
    observation_var = observation_var * ones(1, n_epoch);
else
    observation_var = observation_var(:)';
end

filter_result = multi_model_filter(A_final_turb_mm, acq_times, imm_cfg, observation_var, i_master);
D_candidate_mm = filter_result.phi_seq_mm;
W_imm = compute_imm_confidence(filter_result);

diagnostics = struct( ...
    'model_names', {filter_result.model_stats.names}, ...
    'mean_entropy', filter_result.model_stats.mean_entropy, ...
    'mean_W_imm', mean(W_imm, 'omitnan'), ...
    'parallel_used', filter_result.runtime_stats.parallel_used, ...
    'elapsed_sec', filter_result.runtime_stats.elapsed_sec);

result = struct( ...
    'D_candidate_mm', D_candidate_mm, ...
    'W_imm', W_imm, ...
    'model_probs', filter_result.model_probs, ...
    'model_prob_series', filter_result.model_prob_series, ...
    'model_entropy', filter_result.model_entropy, ...
    'q_final', filter_result.q_final, ...
    'filter_result', filter_result, ...
    'config', imm_cfg, ...
    'diagnostics', diagnostics);
end

function cfg = default_imm_config()
cfg = struct( ...
    'model_type', 'probabilistic', ...
    'fusion_mode', 'imm', ...
    'enabled_models', {{'cv', 'ca'}}, ...
    'q_acc', 2.5e-4, ...
    'R_floor', 9.0, ...
    'adaptive_Q', true, ...
    'Q_adapt_rate', 0.04, ...
    'Q_inflate_max', 3.5, ...
    'innov_window', 9, ...
    'do_rts', true, ...
    'seasonal_period', 365.25, ...
    'huber_k', 1.40, ...
    'outlier_thresh', 4.5, ...
    'student_df', 4.5, ...
    'score_temperature', 1.35, ...
    'imm_self_prob', 0.94, ...
    'model_prior', [0.50, 0.50], ...
    'parallel_enable', false, ...
    'parallel_chunk_size', 7000);
end

function W_imm = compute_imm_confidence(filter_result)
if isfield(filter_result, 'model_entropy') && ~isempty(filter_result.model_entropy)
    entropy = filter_result.model_entropy;
    entropy(~isfinite(entropy)) = 1;
    W_imm = 1 - min(max(entropy, 0), 1);
else
    probs = filter_result.model_probs;
    probs(~isfinite(probs)) = 0;
    W_imm = max(probs, [], 2);
end
W_imm(~isfinite(W_imm)) = 0;
W_imm = min(max(W_imm, 0), 1);
end

function prior = normalize_model_prior(prior)
prior = prior(:)';
if numel(prior) < 2
    prior = [0.5, 0.5];
else
    prior = prior(1:2);
end
if any(~isfinite(prior)) || sum(prior) <= 0
    prior = [0.5, 0.5];
else
    prior = prior / sum(prior);
end
end

function out = merge_struct(base, override)
out = base;
names = fieldnames(override);
for i = 1:numel(names)
    name = names{i};
    if ~isempty(override.(name))
        out.(name) = override.(name);
    end
end
end

function value = get_param(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
