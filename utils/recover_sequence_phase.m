function phi_seq = recover_sequence_phase(phi_ifg, n_epoch, ifg_pairs, i_master)
[n_pts, n_ifg] = size(phi_ifg);
B = zeros(n_ifg, n_epoch);
for k = 1:n_ifg
    B(k, ifg_pairs(k, 1)) = -1;
    B(k, ifg_pairs(k, 2)) = 1;
end
cols = setdiff(1:n_epoch, i_master);
Bsolve = B(:, cols);
phi_seq = nan(n_pts, n_epoch);
for i = 1:n_pts
    obs = phi_ifg(i, :)';
    valid = ~isnan(obs);
    if sum(valid) < max(3, floor(numel(cols) / 3))
        continue;
    end
    sol = Bsolve(valid, :) \ obs(valid);
    row = zeros(1, n_epoch);
    row(cols) = sol;
    row(i_master) = 0;
    phi_seq(i, :) = row;
end
end
