function tf = is_progress_checkpoint(index, total, n_markers)
if nargin < 3 || isempty(n_markers)
    n_markers = 5;
end

if total <= 0
    tf = false;
    return;
end

markers = unique(round(linspace(1, total, min(max(n_markers, 2), total))));
tf = any(index == markers);
end
