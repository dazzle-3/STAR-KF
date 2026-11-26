function out = correlated_residual_filter(e,A1,SigmaY,Kd,params,ref)
% Eqs. (8)-(10): ordered complete-trajectory conditioning, not epoch KF/RTS.
n=numel(e); L=eye(n)-A1; RE=L*SigmaY*L'; RE=(RE+RE')/2;
keep=setdiff(1:n,ref,'stable'); J=eye(n); J=J(:,keep);
[CE,flag]=chol(J'*RE*J,'lower');
if flag~=0
    error('STAR_KF:KF2ReducedRank','KF2 residual covariance is outside reduced-PD domain.');
end
WE=CE\J'; z=WE*e(:); nm=numel(Kd);
scores=zeros(nm,n); Am=zeros(n,n,nm); model=cell(1,nm);
for m=1:nm
    K=L*Kd{m}*L'; K=(K+K')/2; P=K; mu=zeros(n,1); A=zeros(n);
    accepted=false(n-1,1); inflation=ones(n-1,1); innovation=zeros(n-1,1); S0=zeros(n-1,1);
    for j=1:n-1
        h=WE(j,:); v=z(j)-h*mu; S0(j)=1+h*P*h';
        [scores(m,keep(j)),a,u]=student_predictive_score(v,S0(j),params);
        innovation(j)=v; inflation(j)=a;
        if u<=params.outlier_thresh
            g=(P*h')/(h*P*h'+a);
            mu=mu+g*v; A=A+g*(h-h*A);
            P=P-g*(h*P); P=(P+P')/2; accepted(j)=true;
        end
    end
    Am(:,:,m)=A;
    model{m}=struct('Kd',Kd{m},'Kdelta',K,'A2',A,'mean',mu, ...
        'posterior_diagnostic',P,'accepted',accepted,'inflation',inflation, ...
        'Wm',WE(accepted,:),'Gamma',diag(inflation(accepted)), ...
        'innovation',innovation,'S0',S0);
end
[alpha,mu,xi]=predictive_model_weights(scores,params);
A2=zeros(n);
for m=1:nm, A2=A2+alpha(m,:)'.*Am(:,:,m); end
F2=A2*L; C2=F2*SigmaY*F2'; C2=(C2+C2')/2;
out=struct('A2',A2,'F2',F2,'C2',C2,'RE',RE,'WE',WE,'L1',L, ...
    'sequence',(A2*e(:))','variance',max(diag(C2)',0), ...
    'alpha',alpha,'mu',mu,'xi',xi,'scores',scores,'models',{model}, ...
    'whitening_error',norm(WE*RE*WE'-eye(n-1),'fro'));
end
