function [d,support_lost] = frobenius_relative_change(X,Z,mask)
% Eq. (27), no robust clipping or denominator floor. Fixed support only.
if nargin<3, mask=isfinite(Z); end
support_lost=~any(mask(:)) || any(~isfinite(X(mask))) || any(~isfinite(Z(mask)));
if support_lost, d=Inf; return; end
x=X(mask); z=Z(mask); den=max(norm(x),norm(z));
if den==0, d=0; else, d=norm(x-z)/den; end
end
