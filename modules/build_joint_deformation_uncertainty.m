function [marginal,CD,cross] = build_joint_deformation_uncertainty(A1,F2,SigmaY,w)
% Eq. (15b): fixed-fit observation-noise propagation, including shared noise.
cross=A1*SigmaY*F2'; A=A1+w*F2;
CD=A*SigmaY*A'; CD=(CD+CD')/2;
marginal=max(real(diag(CD))',0);
end
