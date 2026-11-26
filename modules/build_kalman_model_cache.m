function cache = build_kalman_model_cache(t_days, params, model)
%BUILD_KALMAN_MODEL_CACHE Precompute time-invariant model matrices.

t_days = t_days(:)';
n = numel(t_days);
model = lower(char(model));
switch model
    case 'cv'
        dim = 2;
    case 'ca'
        dim = 3;
    otherwise
        dim = 4;
end

F = zeros(dim, dim, n);
Q0 = zeros(dim, dim, n);
H = zeros(n, dim);
omega = 2 * pi / max(params.seasonal_period, 1.0);
for k = 1:n
    if k == 1
        dt = 0;
    else
        dt = t_days(k) - t_days(k - 1);
    end
    switch model
        case 'cv'
            F(:, :, k) = [1, dt; 0, 1];
            Q0(:, :, k) = [dt^3/3, dt^2/2; ...
                dt^2/2, dt];
            H(k, :) = [1, 0];
        case 'ca'
            F(:, :, k) = [1, dt, 0.5*dt^2; ...
                0, 1, dt; 0, 0, 1];
            Q0(:, :, k) = [dt^5/20, dt^4/8, dt^3/6; ...
                dt^4/8, dt^3/3, dt^2/2; ...
                dt^3/6, dt^2/2, dt];
            H(k, :) = [1, 0, 0];
        otherwise
            cw = cos(omega * dt);
            sw = sin(omega * dt);
            F(:, :, k) = [1, dt, 0, 0; 0, 1, 0, 0; ...
                0, 0, cw, sw; 0, 0, -sw, cw];
            Q0(:, :, k) = diag([dt^3/3, dt, ...
                0.02*dt, 0.02*dt]);
            H(k, :) = [1, 0, 1, 0];
    end
end

P0=params.initial_covariance.(model);
if ~isequal(size(P0),[dim,dim]) || any(~isfinite(P0(:))) || ...
        norm(P0-P0','fro')>1e-10 || min(eig((P0+P0')/2)) < -1e-10
    error('STAR_KF:InitialPrior','Model initial covariance must be given and PSD.');
end
if n>1 && any(diff(t_days)<=0)
    error('STAR_KF:TimeOrder','Acquisition times must strictly increase.');
end
cache = struct('P0',P0,'model', model, 'n_epoch', n, 'dim', dim, ...
    'F', F, 'Q0', Q0, 'H', H, 'I', eye(dim));
end
