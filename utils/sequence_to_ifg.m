function phi_ifg = sequence_to_ifg(phi_seq_mm, ifg_pairs, scale_mm2rad)
[n_pts, ~] = size(phi_seq_mm);
n_ifg = size(ifg_pairs, 1);
phi_ifg = zeros(n_pts, n_ifg);
for k = 1:n_ifg
    i1 = ifg_pairs(k, 1);
    i2 = ifg_pairs(k, 2);
    phi_ifg(:, k) = (phi_seq_mm(:, i2) - phi_seq_mm(:, i1)) * scale_mm2rad;
end
end
