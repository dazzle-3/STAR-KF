function [B,J,Br,T] = network_incidence(pairs,n,ref)
% Signed IFG incidence and common zero-reference embedding.
B=zeros(size(pairs,1),n);
for k=1:size(pairs,1)
    B(k,pairs(k,1))=-1; B(k,pairs(k,2))=1;
end
keep=setdiff(1:n,ref,'stable');
J=eye(n); J=J(:,keep); Br=B*J;
T=eye(n); T(:,ref)=T(:,ref)-1;
end
