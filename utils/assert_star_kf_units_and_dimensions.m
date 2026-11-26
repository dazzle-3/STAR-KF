function audit = assert_star_kf_units_and_dimensions( ...
    data, stage, varargin)
%ASSERT_STAR_KF_UNITS_AND_DIMENSIONS Validate central STAR-KF contracts.
% Arguments after stage are triples: name, value, contract.

if mod(numel(varargin), 3) ~= 0
    error('Contract arguments must be name/value/contract triples.');
end
i_master = min(max(round(data.i_master), 1), data.n_slc);
audit = struct('stage', char(stage), 'fields', struct());

for k = 1:3:numel(varargin)
    name = char(varargin{k});
    values = varargin{k + 1};
    contract = lower(char(varargin{k + 2}));
    switch contract
        case {'epoch_mm', 'epoch_mm2'}
            expected = [data.n_pts, data.n_slc];
        case {'ifg_rad', 'ifg_rad2'}
            expected = [data.n_pts, data.n_ifg];
        case 'epoch_var_row'
            expected = [1, data.n_slc];
            values = values(:)';
        case 'pixel_weight'
            expected = [data.n_pts, 1];
            values = values(:);
        case {'epoch_cov_mm2','epoch_operator'}
            expected = [data.n_slc,data.n_slc];
        case 'network_operator_mm_per_rad'
            expected = [data.n_slc,data.n_ifg];
        otherwise
            error('Unknown STAR-KF contract: %s.', contract);
    end
    if ~isequal(size(values), expected)
        error('%s at %s has size %s; expected %s for %s.', ...
            name, stage, mat2str(size(values)), mat2str(expected), ...
            contract);
    end
    if any(isinf(values(:)))
        error('%s at %s contains Inf.', name, stage);
    end
    finite_fraction = mean(isfinite(values(:)));
    if finite_fraction <= 0
        error('%s at %s has no finite values.', name, stage);
    end
    if (endsWith(contract, 'mm2') && ~strcmp(contract,'epoch_cov_mm2')) || endsWith(contract, 'rad2') || ...
            strcmp(contract, 'epoch_var_row') || ...
            strcmp(contract, 'pixel_weight')
        finite_values = values(isfinite(values));
        if any(finite_values < -1e-10)
            error('%s at %s contains negative variance/weight values.', ...
                name, stage);
        end
    end
    if strcmp(contract,'epoch_cov_mm2')
        if norm(values-values','fro')>1e-8*max(1,norm(values,'fro')) || ...
                min(eig((values+values')/2)) < -1e-8*max(1,norm(values,'fro')) || ...
                any(abs(values(i_master,:))>1e-7) || any(abs(values(:,i_master))>1e-7)
            error('STAR_KF:CovarianceContract','%s is not symmetric PSD with a zero reference.',name);
        end
    elseif ismember(contract,{'epoch_operator','network_operator_mm_per_rad'})
        if any(abs(values(i_master,:))>1e-7)
            error('STAR_KF:OperatorReference','%s does not preserve the zero reference.',name);
        end
    elseif startsWith(contract, 'epoch_') && ~strcmp(contract, 'epoch_var_row')
        reference_values = values(:, i_master);
        reference_values = reference_values(isfinite(reference_values));
        if ~isempty(reference_values) && ...
                max(abs(reference_values), [], 'omitnan') > 1e-7
            error('%s at %s is not zero at i_master=%d.', ...
                name, stage, i_master);
        end
    end
    audit.fields.(name) = struct( ...
        'contract', contract, ...
        'size', size(values), ...
        'finite_fraction', finite_fraction);
end
end

