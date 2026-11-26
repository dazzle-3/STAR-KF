function result = kalman_filter(z,t_days,r_var,params,model,ref,cache)
% KF1 with fixed-influence RTS operator; no physical covariance floors.
z=z(:)'; n=numel(z); r_var=r_var(:)';
if nargin<7 || isempty(cache), cache=build_kalman_model_cache(t_days,params,model); end
if params.adaptive_Q, error('STAR_KF:ProcessModel','Formal method uses prescribed Q.'); end
d=cache.dim; x=zeros(d,1); P=cache.P0; X=zeros(d,n);
xf=zeros(d,n); Pf=zeros(d,d,n); Xf=zeros(d,n,n);
xp=zeros(d,n); Pp=zeros(d,d,n); Xp=zeros(d,n,n);
score=zeros(1,n); innovation=nan(1,n); S0hist=nan(1,n);
inflation=ones(1,n); accepted=false(1,n); innov_var=nan(1,n);
for k=1:n
    F=cache.F(:,:,k); H=cache.H(k,:);
    if k>1
        x=F*x; X=F*X; P=F*P*F'+params.q_acc*cache.Q0(:,:,k);
    end
    xp(:,k)=x; Pp(:,:,k)=P; Xp(:,:,k)=X;
    if k~=ref && isfinite(z(k)) && isfinite(r_var(k))
        if r_var(k)<0, error('STAR_KF:MeasurementVariance','Negative variance.'); end
        v=z(k)-H*x; S0=H*P*H'+r_var(k);
        [score(k),a,u]=student_predictive_score(v,S0,params);
        innovation(k)=v; S0hist(k)=S0; inflation(k)=a;
        if u<=params.outlier_thresh
            R=a*r_var(k); K=(P*H')/(H*P*H'+R);
            x=x+K*v; basis=zeros(1,n); basis(k)=1;
            X=X+K*(basis-H*X);
            V=eye(d)-K*H; P=V*P*V'+K*R*K';
            accepted(k)=true; innov_var(k)=R;
        end
    end
    P=(P+P')/2; xf(:,k)=x; Pf(:,:,k)=P; Xf(:,:,k)=X;
end
xs=xf; Ps=Pf; Xs=Xf; J=zeros(d,d,max(n-1,0));
if params.do_rts
    for k=n-1:-1:1
        Ppred=Pp(:,:,k+1); A=Pf(:,:,k)*cache.F(:,:,k+1)';
        [C,flag]=chol((Ppred+Ppred')/2,'lower');
        if flag==0, gain=(A/C')/C; else, gain=A*pinv(Ppred); end
        J(:,:,k)=gain;
        xs(:,k)=xf(:,k)+gain*(xs(:,k+1)-xp(:,k+1));
        Xs(:,:,k)=Xf(:,:,k)+gain*(Xs(:,:,k+1)-Xp(:,:,k+1));
        Ps(:,:,k)=Pf(:,:,k)+gain*(Ps(:,:,k+1)-Ppred)*gain';
    end
end
A=zeros(n); seq=zeros(1,n); Kpost=zeros(n);
for k=1:n
    H=cache.H(k,:); seq(k)=H*xs(:,k); A(k,:)=H*Xs(:,:,k);
    Kpost(k,k)=H*Ps(:,:,k)*H';
    if params.do_rts
        G=eye(d);
        for l=k+1:n
            G=G*J(:,:,l-1);
            Kpost(k,l)=H*G*Ps(:,:,l)*cache.H(l,:)'; Kpost(l,k)=Kpost(k,l);
        end
    end
end
T=eye(n); T(:,ref)=T(:,ref)-1;
A=T*A; seq=(T*seq')'; Krel=T*Kpost*T';
result=struct('sequence_mm',seq,'A',A,'state_var_mm2',max(diag(Krel)',0), ...
    'absolute_state_var_mm2',max(diag(Kpost)',0),'q_final',params.q_acc, ...
    'loglike',sum(score),'step_loglike',score,'residual_mm',innovation, ...
    'innov_var',innov_var,'S0',S0hist,'inflation',inflation,'accepted',accepted);
end
