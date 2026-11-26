function pairs = ifgday_to_pairs_idx(acq_times, ifg_times)
acq_times = acq_times(:);
if size(ifg_times, 2) ~= 2
    ifg_times = reshape(ifg_times, [], 2);
end
pairs = zeros(size(ifg_times, 1), 2);
for k = 1:size(ifg_times, 1)
    [~, pairs(k, 1)] = min(abs(acq_times - ifg_times(k, 1)));
    [~, pairs(k, 2)] = min(abs(acq_times - ifg_times(k, 2)));
end
end
