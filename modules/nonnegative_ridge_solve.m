function [x,diagnostic] = nonnegative_ridge_solve(A,b,w,lambda)
% Strict nonnegative weighted ridge by an active-set NNLS algorithm.
% No post-hoc truncation of the unconstrained solution.
n=size(A,2); w=w(:); b=b(:);
H=A'*(w.*A)+lambda*eye(n); f=A'*(w.*b);
x=zeros(n,1); passive=false(n,1); dual=f-H*x;
tol=128*eps*max(1,n)*max(1,norm(f,inf)); steps=0;
while any(~passive & dual>tol)
    candidate=dual; candidate(passive)=-Inf; [~,j]=max(candidate); passive(j)=true;
    while true
        trial=zeros(n,1); trial(passive)=H(passive,passive)\f(passive);
        if all(trial(passive)>0), x=trial; break; end
        hit=passive & trial<=0;
        a=min(x(hit)./(x(hit)-trial(hit)));
        x=x+a*(trial-x); drop=passive & x<=tol; x(drop)=0; passive(drop)=false;
        steps=steps+1;
        if steps>100*n*n, error('STAR_KF:NNLSConvergence','NNLS active set failed.'); end
    end
    dual=f-H*x; steps=steps+1;
end
g=H*x-f;
diagnostic=struct('gradient',g,'weights',w,'solution',x,'active_set_steps',steps, ...
    'objective',sum(w.*(A*x-b).^2)+lambda*sum(x.^2), ...
    'primal_violation',max([0;-x]),'dual_violation',max([0;-g]), ...
    'complementarity',norm(x.*g,inf),'lambda_unnormalized',lambda);
end
