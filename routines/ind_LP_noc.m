function indOut = ind_LP_noc(reg)
% Equation by Equation Local Projection

%% unpack variables
LHS     = reg.LHS;
x       = reg.x;
c       = reg.c;

if isfield(reg,'zc')
    zc = reg.zc;
else
    zc = reg.c;
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

Xall = [x c];
Zall = [zx zc];

%% Estimation
Kall    = size(Xall,2); % K+P
b       = nan(Kall,1,N,H);
se      = nan(Kall,1,N,H);
asymV   = nan(Kall,Kall,N,H);
F       = nan(N,1);


for i = 1:N
    Y    = LHS(T*(i-1)+1:T*i,:);
    X    = Xall(T*(i-1)+1:T*i,:);
    Z    = Zall(T*(i-1)+1:T*i,:);
    
    % 2SLS
    Xhat = Z*((Z'*Z)\(Z'*X)); % first stage
    bhat = (Xhat'*Xhat)\(Xhat'*Y);
    ehat = Y - X*bhat;
    b(:,:,i,:) = bhat;
    
    % compute HAC SE
    for h = 1:H
        g              = Xhat.*repmat(ehat(:,h),1,size(X,2));
        v_hac          = HAC(g,nwtrunc);
        vbeta          = (Xhat'*Xhat)\v_hac/(Xhat'*Xhat);
        se(:,:,i,h)    = sqrt(diag(vbeta));
        asymV(:,:,i,h) = vbeta;
    end
    
    % Compute F-statistics
    X     = X(:,K);
    gam   = (Z'*Z)\(Z'*X);
    uhat  = X-Z*gam;
    g     = Z.*repmat(uhat,1,size(Z,2));
    v_hac = HAC(g,H+1);
    vbeta = (Z'*Z)\v_hac/(Z'*Z);
    tmp   = sqrt(diag(vbeta));
    F(i)  = (gam(1)/tmp(1))^2;
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
indOut.F     = F;
end
