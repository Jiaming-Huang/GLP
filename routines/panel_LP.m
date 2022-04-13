function panOut = panel_LP(reg, FE)
% Standard Panel Local Projection (Pooled)
% b_hat =  [ (\sum_i Xi'Z_i)S(\sum_i Zi'Xi) ]\ [ (\sum_i Xi'Z_i)S(\sum_i Zi'Yi) ]
% reference: Jorda and Taylor EJ (2016)

% Input
% reg contains x y (z) LHS control and params
% FE = 0, first difference (OLS) the data should be already differenced!!
%    = 1, fixed effects (demean)
%    = 2, random effects (OLS)
% difference b/w FD and RE is that FD is estimated without constant term
% workflow:
% prepare the data -> OLS/2SLS -> compute HAC -> (compute First stage F)

% the program has been tested using stata xtivreg2, fe cluster(id)
% see the stata file in output\APPEN\compare_panel_lp

%% unpack variables
LHS     = reg.LHS;
x       = reg.x;
c       = reg.c;

if isfield(reg,'zx')
    zx = reg.zx;
else
    zx = reg.x;
end
if isfield(reg,'zc')
    zc = reg.zc;
else
    zc = reg.c;
end

N       = reg.param.N;
T       = reg.param.T;
H       = size(LHS,2);
K       = size(x,2);
P       = size(c,2);

%% Preprocessing
if FE == 1
    % fixed effects: demean
    tmp    = [LHS x c zx zc];
    tmp_de = gdemean(tmp,N,T);
    LHS    = tmp_de(:,1:H);
    X      = tmp_de(:,H+1:H+K+P);
    Z      = tmp_de(:,H+K+P+1:end);
elseif FE == 0
    % random effects: add constant to c
    X = [x c ones(N*T,1)];
    Z = [zx zc ones(N*T,1)];
else
    % FD: we require all variables have been differenced
    X = [x c];
    Z = [zx zc];
end

Kall    = size(X,2);

%% Estimation: 2SLS
S      = inv(Z'*Z);
bhat   = (X'*Z*S*Z'*X)\(X'*Z*S*Z'*LHS);
e      = LHS - X*bhat;
se     = nan(Kall,H);
asymV  = nan(Kall,Kall,H);

% Cameron and Miller - JHR (2015)
% see equation (30)
Qxx  = (X'*Z*S*Z'*X);
for h = 1:H
    % cluster at individual level
    Zep           = Z.*e(:,h);
    Sigma         = X'*Z*S*panel_HAC(Zep,N,T)*S*Z'*X;
    vbeta         = Qxx\Sigma/Qxx;
    se(:,h)       = sqrt(diag(vbeta));
    asymV(:,:,h)  = vbeta;
end

%% Store output
IR            = nan(K,1,1,H);
IR(:,:,1,:)   = bhat(1:K,:);
IRse          = nan(K,1,1,H);
IRse(:,:,1,:) = se(1:K,:);
IRUb          = IR + 1.96*IRse;
IRLb          = IR - 1.96*IRse;

panOut.IR     = IR;
panOut.IRse   = IRse;
panOut.IRUb   = IRUb;
panOut.IRLb   = IRLb;
panOut.b      = bhat;
panOut.se     = se;
panOut.asymV  = asymV;

end

function Sigma = panel_HAC(Xu,N,T)
%{

Sigma estimator, clustered at individual level.

The general formula is:

sum_g=1^G  X_g'*u_g*u_g'*X_g

and now G=N.

Input: Xu, NT x P matrix of X.*uhat
          in the case of GMM, it is Z.*uhat where uhat is the residual
        
Output: Sigma^{cluster} (not scaled by NT)

Reference:
[1] SW, Econometrica 2008, compute Sigma^{cluster}, equation (10)
[2] Cameron and Miller, JHR 2015, equation (11) and (30)
%}
P = size(Xu,2);
if N> 1
    Xu = reshape(Xu',P,T,N);              % Now P by T by N matrix
    tmp = reshape(sum(Xu,2),P,N);        % sum across T, now P by N matrix
else
    tmp = Xu';
end
Sigma = tmp*tmp';

end
