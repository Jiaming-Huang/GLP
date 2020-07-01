function [b, e, se] = panel_LP(reg, FE)
% Carry out panel Local Projection
% reg contains x y (z) LHS control and params
% workflow:
% demean -> OLS/2SLS -> compute HAC -> (compute First stage F)

%% unpack variables
LHS = reg.LHS;
x   = reg.x;
z   = reg.z;
control = reg.control;
N = reg.param.N;
T = reg.param.T;
H = size(LHS,2);



if isempty(z)
    fprintf('Model: Panel Local Projection (without IV)\n')
    
    if FE == 1
    %===================== Step 1 : demean =======================%
    tmp    = [LHS x control];
    tmp_de = gdemean(tmp,reg.param);
    LHS    = tmp_de(:,1:H);
    x      = tmp_de(:,H+1);
    if isempty(control)
        control = [];
    else
        control= tmp_de(:,H+2:end);
    end
    end
    %===================== Step 2 : OLS =======================%
    % project out controls
    LHS = LHS - control*((control'*control)\(control'*LHS));
    X = x - control*((control'*control)\(control'*x));
    XX = sum(mean(reshape(X.^2,T,N)));
    Xy = sum(reshape(mean(reshape(X.*LHS,T,N*H)),N,H));
    bhat = (XX)\(Xy);
    e    = LHS - X*bhat;
    b    = bhat(1,:);
    
    %===================== Step 3 : Compute HAC =======================%
    % Stock and Watson, Econometrica 2008, Sigma_cluster
    se   = nan(1,H);
    Qxx  = (X'*X)/N/T;
    
    for h = 1:H
        g     = X.*repmat(e(:,h),1,size(X,2));
        Sigma = panel_HAC(g,N,T);
        vbeta = Qxx\Sigma/Qxx/N/T;
        tmp   = sqrt(diag(vbeta));
        se(h) = tmp(1);
    end
    
    
    
else
    %fprintf('Model: Panel Local Projection (with IV)\n')
    if FE==1
    %===================== Step 1 : demean =======================%
    tmp    = [LHS x control];
    tmp_de = gdemean(tmp,reg.param);
    LHS    = tmp_de(:,1:H);
    x      = tmp_de(:,H+1);
    if isempty(control)
        control = [];
    else
        control= tmp_de(:,H+2:end);
    end
    end
    
    %===================== Step 2 : First Stage =======================%
    % project out controls
    LHS = LHS - control*((control'*control)\(control'*LHS));
    X = x - control*((control'*control)\(control'*x));
    Z = z - control*((control'*control)\(control'*z));  
    Omega = 1./mean(reshape(Z.^2,T,N));
    zx   = mean(reshape(Z.*X,T,N));
    zxoxz = (zx.*Omega)*zx';
    zy    = reshape(mean(reshape(Z.*LHS,T,N*H)),N,H);
    zxozy = (zx.*Omega)*zy;
    b = (zxoxz\zxozy);
    e    = LHS - X*b;
    
    %===================== Step 4 : Compute HAC =======================%
    % Stock and Watson, Econometrica 2008, Sigma_cluster
    Qxx  = zxoxz/N;
    zxo  = (zx.*Omega);
    PHI  = nan(1,H);
    for h = 1:H
        % not clustered at individual level
%         zep = reshape(Z.*e(:,h),T,N);
%         PHI(h) = zxo*(zep'*zep/T)*zxo'/N;
        % cluster at individual level
        zep = reshape((Z.*e(:,h)).^2,T,N);
        zep = mean(zep);
        PHI(h) = zxo.^2*zep'/N;
    end
    se = sqrt(inv(Qxx)*PHI*inv(Qxx)/N/T);

end


end

function x_de = gdemean(x,param)
% demeaning: remove unit fixed effects
x_de = nan(size(x,1),size(x,2));
for i = 1: param.N
    % for each unit
    xi = x(param.T*(i-1)+1:param.T*i,:);
    x_mean = mean(xi);
    x_de(param.T*(i-1)+1:param.T*i,:) = xi - repmat(x_mean,param.T,1);
end
end

function Sigma = panel_HAC(g,N,T)
%{

Based on SW, Econometrica 2008, compute Sigma^{cluster}

Input: g, NT x P matrix of X.*ehat

Output: Sigma^{cluster}

%}
P = size(g,2);
if N> 1
g = reshape(g',P,T,N);              % Now P by T by N matrix
tmp = reshape(sum(g,2),P,N);        % sum across T, now P by N matrix
else
    tmp = g';
end
Sigma = tmp*tmp'/N/T;

end
