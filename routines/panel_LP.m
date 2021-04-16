function [b, se] = panel_LP(reg, method)
% Standard Panel Local Projection (Pooled)
% reference: Jorda and Taylor EJ (2016)

% Input
% reg contains x y (z) LHS control and params
% method = 0, random effects (OLS)
%        = 1, fixed effects (demean)
% workflow:
% prepare the data -> OLS/2SLS -> compute HAC -> (compute First stage F)

% the program has been tested using stata xtivreg2, fe cluster(id)
% see the stata file in output\APPEN\compare_panel_lp

%% unpack variables
LHS = reg.LHS;
X   = reg.x;
Z   = reg.z;
control = reg.control;

N = reg.param.N;
T = reg.param.T;
H = size(LHS,2);
P = size(control,2);


if isempty(Z)
    %     fprintf('Model: Panel Local Projection (without IV)\n')
    if method == 1
        %===================== Step 1 : demean =======================%
        tmp    = [LHS X control];
        tmp_de = gdemean(tmp,N,T);
        LHS    = tmp_de(:,1:H);
        X      = tmp_de(:,H+1:H+1);
        if P~=0
            control = tmp_de(:,H+2:H+1+P);
            X = [X control];
        end
    end
    %===================== Step 2 : OLS =======================%
    bhat = (X'*X)\(X'*LHS);
    e    = LHS - X*bhat;
    b    = bhat(1,:);
    
    %=============== Step 3 : Robust Standard Errors =================%
    se   = nan(1,H);
    Qxx  = (X'*X);
    for h = 1:H
        g     = X.*repmat(e(:,h),1,size(X,2));
        Sigma = panel_HAC(g,N,T);
        vbeta = Qxx\Sigma/Qxx;
        tmp   = sqrt(diag(vbeta));
        se(h) = tmp(1);
    end
    
    
else
    %fprintf('Model: Panel Local Projection (with IV)\n')
    if method==1
        %===================== Step 1 : demean =======================%
        tmp    = [LHS X control];
        tmp_de = gdemean(tmp,N,T);
        LHS    = tmp_de(:,1:H);
        X      = tmp_de(:,H+1);
        if P~=0
            control = tmp_de(:,H+2:H+1+P);
        end
    end
    Z = [Z control];
    X = [X control];
    %=====================   Step 2 : TSLS   =======================%
    W = inv(Z'*Z);
    bhat = (X'*Z*W*Z'*X)\(X'*Z*W*Z'*LHS);
    e    = LHS - X*bhat;
    b = bhat(1,:);
    
    %=============== Step 3 : Robust Standard Errors =================%
    % Cameron and Miller - JHR (2015)
    % see equation (30)
    se   = nan(1,H);
    Qxx  = (X'*Z*W*Z'*X);
    for h = 1:H
        % cluster at individual level
        Zep = Z.*e(:,h);
        Sigma = X'*Z*W*panel_HAC(Zep,N,T)*W*Z'*X;
        vbeta = Qxx\Sigma/Qxx;
        tmp = sqrt(diag(vbeta));
        se(h) = tmp(1);
    end
end


end

function x_de = gdemean(x,N,T)
% designed for balanced panel
x_de = nan(size(x,1),size(x,2));
for i = 1: N
    % for each unit
    xi = x(T*(i-1)+1:T*i,:);
    x_mean = mean(xi);
    x_de(T*(i-1)+1:T*i,:) = xi - repmat(x_mean,T,1);
end
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
