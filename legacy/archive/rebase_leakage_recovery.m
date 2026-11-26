function [D_recovered_v2_mm, D_leak_mm] = rebase_leakage_recovery( ...
    D_filter_mm, D_corrected_mm, D_recovered_mm, gamma)
if nargin < 4 || isempty(gamma)
    gamma = 1.0;
end

validateattributes(gamma, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'nonnegative'});
if ~isequal(size(D_filter_mm), size(D_corrected_mm), ...
        size(D_recovered_mm))
    error(['D_filter_mm, D_corrected_mm, and D_recovered_mm ', ...
        'must have the same size.']);
end

D_leak_mm = D_recovered_mm - D_filter_mm;
D_recovered_v2_mm = D_corrected_mm + gamma .* D_leak_mm;
end
