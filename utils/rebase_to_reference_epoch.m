function values = rebase_to_reference_epoch(values, i_master)
%REBASE_TO_REFERENCE_EPOCH Set a time-series reference epoch to zero.

if ~ismatrix(values) || isempty(values)
    error('values must be a non-empty two-dimensional array.');
end
i_master = min(max(round(i_master), 1), size(values, 2));
reference = values(:, i_master);
values = values - reference;
values(:, i_master) = 0;
end

