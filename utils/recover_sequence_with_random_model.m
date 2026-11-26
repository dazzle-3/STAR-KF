function [seq,operators] = recover_sequence_with_random_model(phi,data,ref,R,components,opt)
% Formal Eq. (2)-(3); historical random-model flags cannot alter this path.
if ~opt.network_dyy
    error('STAR_KF:FormalCovariance','Target method requires full network covariance.');
end
model=build_ifg_covariance_model(data.ifg_pairs,data.n_slc,ref, ...
    R/data.scale_rad2mm^2,components.ifg_extra_rad2,opt.dyy_reg_fraction);
[seq,operators]=recover_sequence_phase_weighted(phi,data.n_slc, ...
    data.ifg_pairs,ref,model,opt.recovery_lambda);
operators.G=operators.Gphase*data.scale_rad2mm;
operators.SigmaY=operators.SigmaPhase*data.scale_rad2mm^2;
operators.SigmaPhi=model.Dyy; operators.QI=model.QI;
operators.covariance_semantics='unloaded physical covariance through fixed loaded operator';
end
