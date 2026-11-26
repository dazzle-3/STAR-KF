function [updated_r, components] = uncertainty_propagation(phi_turb_var_ifg, sigma2_slc, ifg_pairs, filter_var_mm2, model_entropy, residual_ifg_rad, scale_rad2mm, jcfg, r_floor, ref_epoch)
if nargin < 10, ref_epoch = 1; end
scale_sq = scale_rad2mm^2;
ifg_turb_var_rad2 = mean(phi_turb_var_ifg, 1, 'omitnan');
ifg_resid_var_rad2 = mean(residual_ifg_rad.^2, 1, 'omitnan');
ifg_turb_var = ifg_turb_var_rad2 * scale_sq;
ifg_resid_var = ifg_resid_var_rad2 * scale_sq;
slc_turb = propagate_ifg_to_slc(ifg_turb_var, ifg_pairs, numel(sigma2_slc), ref_epoch);
slc_resid = propagate_ifg_to_slc(ifg_resid_var, ifg_pairs, numel(sigma2_slc), ref_epoch);
filter_epoch_var = mean(filter_var_mm2, 1, 'omitnan');
mean_model_entropy = mean(model_entropy, 'omitnan');
model_term = mean_model_entropy * max(filter_epoch_var, 1.0);
posterior = sigma2_slc * scale_sq + slc_turb + jcfg.residual_weight * slc_resid + ...
    jcfg.deformation_prior_weight * filter_epoch_var + jcfg.model_weight * model_term;
updated_r = max(posterior, max(r_floor, jcfg.lower_bound_mm2));
ifg_white_var_rad2 = sigma2_slc(ifg_pairs(:, 1)) + sigma2_slc(ifg_pairs(:, 2));
ifg_white_var_mm2 = ifg_white_var_rad2(:)' * scale_sq;
ifg_extra_rad2 = ifg_turb_var_rad2 + jcfg.residual_weight * ifg_resid_var_rad2;
ifg_total_rad2 = ifg_white_var_rad2(:)' + ifg_extra_rad2;
ifg_total_mm2 = ifg_total_rad2 * scale_sq;
components = struct('slc_turbulence_mm2', slc_turb, 'slc_residual_mm2', slc_resid, ...
    'slc_filter_mm2', filter_epoch_var, 'slc_model_mm2', model_term, ...
    'slc_white_mm2', sigma2_slc * scale_sq, ...
    'ifg_turbulence_rad2', ifg_turb_var_rad2, 'ifg_residual_rad2', ifg_resid_var_rad2, ...
    'ifg_white_rad2', ifg_white_var_rad2(:)', 'ifg_extra_rad2', ifg_extra_rad2, ...
    'ifg_total_rad2', ifg_total_rad2, 'ifg_turbulence_mm2', ifg_turb_var, ...
    'ifg_residual_mm2', ifg_resid_var, 'ifg_white_mm2', ifg_white_var_mm2, ...
    'ifg_total_mm2', ifg_total_mm2);
end

function slc_var = propagate_ifg_to_slc(ifg_var, ifg_pairs, n_slc, ref_epoch)
keep = setdiff(1:n_slc, ref_epoch);
B = zeros(numel(ifg_var), numel(keep));
for k = 1:size(ifg_pairs,1)
    i = ifg_pairs(k,1); j = ifg_pairs(k,2);
    ii = find(keep==i,1); jj = find(keep==j,1);
    if ~isempty(ii), B(k,ii) = -1; end
    if ~isempty(jj), B(k,jj) = 1; end
end
valid = isfinite(ifg_var) & ifg_var > 0;
if nnz(valid) < max(3, floor(size(B,2)/2))
    fill = median(ifg_var(valid)); if ~isfinite(fill), fill = 25; end
    slc_var = fill * ones(1,n_slc); slc_var(ref_epoch) = min(fill, 10); return;
end
W = diag(1 ./ max(ifg_var(valid), 1e-6));
N = B(valid,:)' * W * B(valid,:);
Cov = pinv(N);
slc_var = nan(1,n_slc); slc_var(ref_epoch) = min(median(diag(Cov),'omitnan'), 10);
for i = 1:numel(keep), slc_var(keep(i)) = Cov(i,i); end
slc_var(~isfinite(slc_var)) = median(slc_var(isfinite(slc_var)));
slc_var = max(slc_var, 1e-6);
end
