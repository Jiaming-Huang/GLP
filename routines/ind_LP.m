function indOut = ind_LP(reg)
% Equation by Equation Local Projection

%% unpack variables
LHS     = reg.LHS;
x       = reg.x;

if isfield(reg,'c')
    c = reg.c;
    if isfield(reg,'zc')
        zc = reg.zc;
    else
        zc = reg.c;
    end
else
    c = [];
    zc = [];
end
if isfield(reg,'zx')
    zx = reg.zx;
else
    zx = reg.x;
end

N       = reg.param.N;
T       = reg.param.T;
H       = size(LHS,2);
K       = size(x,2);
nwtrunc = reg.param.nwtrunc;

Xall = [x c ones(N*T,1)];
Zall = [zx zc ones(N*T,1)];

%% Estimation
Kall    = size(Xall,2); % K+P+1
Lall    = size(Zall,2);
b       = nan(Kall,1,N,H);
se      = nan(Kall,1,N,H);
asymV   = nan(Kall,Kall,N,H);
v_hac   = nan(Lall,Lall,N,H);
F       = nan(N,1);
res     = nan(N*T,H);

for i = 1:N
    Y    = LHS(T*(i-1)+1:T*i,:);
    X    = Xall(T*(i-1)+1:T*i,:);
    Z    = Zall(T*(i-1)+1:T*i,:);

    % 2SLS
    % first stage
    Pi    = (Z'*Z)\(Z'*X);
    uhat  = X(:,1:K)-Z*Pi(:,1:K);
    g     = Z.*repmat(uhat,1,size(Z,2));
    v     = HAC(g,H+1);
    vbeta = (Z'*Z)\v/(Z'*Z);
    tmp   = sqrt(diag(vbeta));
    F(i)  = (Pi(1)/tmp(1))^2;

    % second stage
    Xhat = Z*Pi; % first stage
    bhat = (Xhat'*Xhat)\(Xhat'*Y);
    ehat = Y - X*bhat;
    b(:,:,i,:) = bhat;
    res(T*(i-1)+1:T*i,:) = ehat;

    % compute HAC SE
    for h = 1:H
        g              = Z.*repmat(ehat(:,h),1,size(X,2));
        v_hac(:,:,i,h) = HAC(g,nwtrunc);
        vbeta          = (Xhat'*Xhat)\Pi'*v_hac(:,:,i,h)*Pi/(Xhat'*Xhat);
        se(:,:,i,h)    = sqrt(diag(vbeta));
        asymV(:,:,i,h) = vbeta;
    end

    
end

%% Store output
IR           = b(1:K,:,:,:);
IRse         = se(1:K,:,:,:);
IRUb         = IR + 1.96*IRse;
IRLb         = IR - 1.96*IRse;

indOut.IR    = IR;
indOut.IRUb  = IRUb;
indOut.IRLb  = IRLb;
indOut.IRse  = IRse;
indOut.b     = b;
indOut.se    = se;
indOut.asymV = asymV;
indOut.v_hac = v_hac;
indOut.F     = F;
indOut.res   = res;
end
