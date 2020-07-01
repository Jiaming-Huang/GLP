function [Group, GIRF, GSE, Qpath, gpath, bpath] = GroupLPIV_TrueIRF(reg, K, bTrue, tsls, FE)
% heterogeneous coef of controls
% written for the case with scalar x and scalar z

% INPUT:
%   reg: data (reg.x, reg.y, reg.z, reg.LHS, reg.control, reg.param)
%   K: number of groups to be classified
%   ninit: number of initializations
%   2sls: 1 if 2sls; by default 0 (IV estimator)

% OUTPUT:
%   Group: Group composition, N by 1 vector
%   GIRF: Group IRF, K by H+1 matrix
%   Qpath: path of LOSS
%   gpath: path of groups
%   bpath: path of coef (could be super large) K by h by ninit

if nargin < 4
    tsls = 0;
end

if nargin < 5
    FE = 1;
end



%% UNPACK VARIABLES
N = reg.param.N;
T = reg.param.T;

% construct regression variables
LHS = reg.LHS;
x   = reg.x;
z   = reg.z;
control = reg.control;

H = size(LHS,2);

%% PREPARE VARIABLES
GIRF = nan(K,H);
GSE = nan(K,H);

if isempty(control)
    if FE == 1
    tmp    = [LHS x];
    tmp_de = gdemean(tmp,reg.param);
    y      = tmp_de(:,1:H);
    x_p    = tmp_de(:,H+1);
    else
        y = LHS;
        x_p = x;
    end
    z_p = z;
    
else
    if FE == 1
    % demean
    tmp    = [LHS x control];
    tmp_de = gdemean(tmp,reg.param);
    LHS    = tmp_de(:,1:H);
    x      = tmp_de(:,H+1);
    control= tmp_de(:,H+2:end);
    end
    % Partial out controls ===> x_p and z_p are all NT by 1 vectors!
    for i = 1: N
        ytmp = LHS(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        ztmp = z(T*(i-1)+1:T*i,:);
        wtmp = control(T*(i-1)+1:T*i,:);
        
        % partial out controls
        y(T*(i-1)+1:T*i,:) = ytmp - wtmp*((wtmp'*wtmp)\(wtmp'*ytmp));
        x_p(T*(i-1)+1:T*i,:) = xtmp - wtmp*((wtmp'*wtmp)\(wtmp'*xtmp));
        z_p(T*(i-1)+1:T*i,:) = ztmp - wtmp*((wtmp'*wtmp)\(wtmp'*ztmp));
    end
    
end

% concated version of y and x
YY = reshape(y,T*N*H,1);
XX = kron(eye(H),x_p);
ZZ = kron(ones(H,1),z_p);




%% ESTIMATION SETTING

tol = 1e-6;
nIter = 100;  % by default, number of iteration should be less than 100

Qpath = nan(nIter,1);
bpath = cell(nIter,1);
gpath = nan(N,nIter);


if tsls == 1  % 2 stage least squares
    % get Weighting matrix Omega_i = (sum_t z_{i,t}^2 / T)^{-1}
    Omega = 1 ./ mean(reshape(z_p.^2,T,N));
    Omega_assign = repmat(Omega,1,H);
else
    Omega = ones(1,N);
    Omega_assign = ones(1,N*H);
end


%% ESTIMATION

if K == 1   % Trivial case, single group
    zx   = mean(reshape(z_p.*x_p,T,N));
    zxoxz = (zx.*Omega)*zx';
    zy    = reshape(mean(reshape(z_p.*y,T,N*H)),N,H);
    zxozy = (zx.*Omega)*zy;
    b = (zxoxz\zxozy);
    m = ZZ.*(YY - XX*b');           % the loss function is scaled by Omega. So 2sls is the same as iv with unit-invariant z
    m = mean(reshape(m,T,N*H)) .^2;
    Qpath = mean(sum(reshape(Omega_assign.*m,N,H),2));
    Group = ones(N,1);
    %GIRF = b;
    gpath = []; bpath = [];
    
else % main algorithm starts here
    b_old = bTrue;
    b_new = nan(K,size(y,2));
    Q_old = 999;
    
    for iter = 1:nIter % iterate over assignment and updating
        
        %% assignment step
        di = nan(N, K); % store the Euclidean distance b/w y and Xb for each b
        for k = 1:K
            mm = ZZ.*(YY-XX*b_old(k,:)');
            mm = sum(reshape(mm,T,N*H));    % should divided by T, but just scaling
            di(:,k) = sum(reshape(Omega_assign .*(mm.^2),N,H),2);
        end
        [~,g] = min(di,[],2);      
        
        uniquek = unique(g);
        if length(uniquek) < K  % empty group
            lucky = randperm(N);
            for l = setdiff(1:K, uniquek)
                g(lucky(l)) = l;
            end
        end
        
        %% update step
        G = kron(g,ones(T,1));
        obj = zeros(K,1);
        for k = 1:K
            Nk = sum(g==k);
            ytmp = y(G==k,:);
            xtmp = x_p(G==k,:);
            ztmp = z_p(G==k,:);
            Omega_tmp = Omega(g==k);
            zx   = mean(reshape(ztmp.*xtmp,T,Nk));
            zxoxz = (zx.*Omega_tmp)*zx';
            zy    = reshape(mean(reshape(ztmp.*ytmp,T,Nk*H)),Nk,H);
            zxozy = (zx.*Omega_tmp)*zy;
            b_new(k,:) = (zxoxz\zxozy);
            m = ztmp.*(ytmp - xtmp*b_new(k,:));
            m = reshape(m,T,size(m,1)/T,H);
            obj(k) = sum(sum(Omega_tmp .*(mean(m).^2),3));
        end
        
        Q_new = sum(obj)/N;
        
        % check convergence
        [dif, d] = resid(b_old,b_new,Q_old,Q_new,tol);
        
        fprintf('Iteration %d: Max diff %f \n', iter, dif)
        if d == 1
            break;
        end
        
        Q_old = Q_new;
        b_old = b_new;
    end
    Qpath(iter) = Q_old;
    bpath{iter} = b_old;
    gpath(:,iter) = g;
    
    %% Final Step: find minimal initialization
    [~,Qid] = min(Qpath);
    Group = gpath(:,Qid);
    %GIRF = bpath{Qid};
end


%% Inference


%% Large T inference
% now the IRFs could be a bit different
% 1) we allow within group controls to affect individuals
% 2) panel_LP is the pooled GMM
% 3) panel_LP1 is the individual GMM in the paper
% gtmp = kron(Group,ones(T,1));
% tmp = [];
% for k = 1:K
%     tmp.y = reg.y(gtmp==k,:);
%     tmp.x = reg.x(gtmp==k,:);
%     tmp.z = reg.z(gtmp==k,:);
%     tmp.LHS = reg.LHS(gtmp==k,:);%cumsum(reg.LHS(gtmp==k,:),2);
%     tmp.control = reg.control(gtmp==k,:);
%     tmp.param.N = sum(Group==k);
%     tmp.param.T = reg.param.T;
%     [GIRF(k,:), ~, GSE(k,:)] = panel_LP1(tmp,FE);
% end

%% Fixed T inference
% get large T Gamma and V
% correction for mis-classification
Gam = zeros(K,K,H);
V   = zeros(K,K,H);
NK  = zeros(K,K);
g   = kron(Group,ones(T,1));

% prepare variables
LHS = reg.LHS;
x   = reg.x;
z   = reg.z;
control = reg.control;
if FE==1
    %===================== Step 1 : demean =======================%
    tmp    = [LHS x control];
    tmp_de = gdemean(tmp,reg.param);
    y    = tmp_de(:,1:H);
    x      = tmp_de(:,H+1);
    if isempty(control)
        control = [];
    else
        control= tmp_de(:,H+2:end);
    end
end

for k = 1:K
    Nk = sum(Group==k);
    NK(k,k) = Nk;
    ytmp = y(g==k,:);
    xtmp = x(g==k,:);
    ztmp = z(g==k,:);
    ctmp = control(g==k,:);
    % partial out controls
    ytmp = ytmp - ctmp*((ctmp'*ctmp)\(ctmp'*ytmp));
    xtmp = xtmp - ctmp*((ctmp'*ctmp)\(ctmp'*xtmp));
    ztmp = ztmp - ctmp*((ctmp'*ctmp)\(ctmp'*ztmp));
    % store the "cleaned" ones for later use
    y(g==k,:)   = ytmp;
    x_p(g==k,:) = xtmp;
    z_p(g==k,:) = ztmp;
    
    % compute Gam and V
    Omega      = 1./mean(reshape(ztmp.^2,T,Nk));
    zx         = sum(reshape(ztmp.*xtmp,T,Nk));
    zxoxz      = (zx.*Omega)*zx';
    Gam(k,k,:) = zxoxz/N;
    zy         = reshape(sum(reshape(ztmp.*ytmp,T,Nk*H)),Nk,H);
    zxozy      = (zx.*Omega)*zy;
    GIRF(k,:)  = (zxoxz\zxozy);
    e          = ytmp - xtmp*GIRF(k,:);
    
    for h = 1:H
        zep      = sum(reshape((ztmp.*e(:,h)).^2,T,Nk));
        V(k,k,h) = (zx.*Omega).^2*zep'/N;
    end
end

Omega = 1./mean(reshape(z_p.^2,T,N));
zxo = sum(reshape(z_p.*x_p,T,N)).*Omega;
% correction
for i = 1:K
    for j = 1:K
        for h = 1:H
            % correct position Gam(i,j,h)
            if i == j %diagonal element
                cor = 0;
                termA = reshape(sum(reshape(z_p.*(y - x_p.*GIRF(i,:)),T,N*H)),N,H).*zxo';
                termB = termA(:,h);
                termA = sum(termA,2);
                for l = 1:K
                    if l ~= i
                        termC = sum((sum(reshape(z_p,T,N)).*zxo)'.*abs(GIRF(l,:)-GIRF(i,:)),2);
                        % compute density
                        dist = sum(reshape(sum(reshape(z_p.*(y-x_p.*GIRF(l,:)),T,N*H)),N,H).^2.*Omega'-...
                            reshape(sum(reshape(z_p.*(y-x_p.*GIRF(i,:)),T,N*H)),N,H).^2.*Omega',2);
                        bw   = std(dist)*(4/3/N)^(0.2);
                        den  = normpdf(dist/bw)/bw .* (Group == l | Group ==i);
                        cor  = cor + sum(termA.*termB.*den./termC);
                    end
                end
                Gam(i,j,h) = Gam(i,j,h)-cor/N;
            else
                termA = sum(reshape(sum(reshape(z_p.*(y - x_p.*GIRF(j,:)),T,N*H)),N,H).*zxo',2);
                termB = (sum(reshape(z_p.*(y(:,h) - x_p.*GIRF(i,h)),T,N)).*zxo)';
                termC = sum((sum(reshape(z_p,T,N)).*zxo)'.*abs(GIRF(j,:)-GIRF(i,:)),2);
                dist  = sum(reshape(sum(reshape(z_p.*(y-x_p.*GIRF(j,:)),T,N*H)),N,H).^2.*Omega'-...
                            reshape(sum(reshape(z_p.*(y-x_p.*GIRF(i,:)),T,N*H)),N,H).^2.*Omega',2);
                bw    = std(dist)*(4/3/N)^(0.2);
                den   = normpdf(dist/bw)/bw .* (Group == j | Group ==i);
                Gam(i,j,h) = Gam(i,j,h)+ mean(termA.*termB.*den./termC);
            end
        end
    end
end
    
for h = 1:H
    GSE(:,h) = diag(sqrt(inv(Gam(:,:,h))*V(:,:,h)*inv(Gam(:,:,h))./NK));
end

end

function x_de = gdemean(x,param)
x_de = nan(size(x,1),size(x,2));
for i = 1: param.N
    % for each unit
    xi = x(param.T*(i-1)+1:param.T*i,:);
    x_mean = mean(xi);
    x_de(param.T*(i-1)+1:param.T*i,:) = xi - repmat(x_mean,param.T,1);
end
end