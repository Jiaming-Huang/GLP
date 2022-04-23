function [Gr_EST, GIRF, OBJ, IC] = GLP_SIM_UnknownG0(reg, Gmax, nInit, bInit, weight, FE)
% GLP algorithm used for simulation
% The only difference b/w this function and GLP.m is that we do inference here

% Related functions:
% GLP_SIM_KnownG0.m: this function takes G0 as given and initializes with true IRs and just run one iteration
% GLP_SIM_Infeasible.m: this function runs infeasible GLP (with fully pooled panel LP-IV for each group)
% GLP.m: this is a fully fledged version of GLP that can be used for empirical applications

% --------------------------- INPUT --------------------------------
%   reg: data (reg.LHS, reg.x, reg.c, reg.zx, reg.zc, reg.param)
%           reg.LHS NT by H dependent variables
%           reg.x, NT by K policy variables whose coefs are to be grouped
%           reg.zx, NT by Lx IV for reg.x (optional, use reg.x if not specified)
%           reg.c, NT by P controls whose coefs vary across i (can be empty)
%           reg.zc NT by Lc IV for reg.c (optional, use reg.c if not specified)
%           reg.param.N, reg.param.T
%           for simplicity, the program is written for balanced panel, but can be easily adapted to unbalanced ones
%   Gmax: number of groups to be classified
%   nInit: number of initializations
%   bInit: potential initial values, see below
%   weight: either string ('2SLS', 'IV') or user-supplied weights;
%   FE: 1 - fixed effects (within estimator, demean)

% --------------------------- OUTPUT --------------------------------
%   Gr_EST: Group composition, N by Gmax matrix
%   GIRF: Group IRF, 1 by Gmax cell, with K by 1 by G by H coefs
%   OBJ: minimized objective function for each Ghat, 1 by Gmax vector
%   IC: Group IRF, K by H by Gmax matrix

% when weight, FE, large T are not supplied
% by default we use:
% 1) inverse of asym.variance as weighting matrix
% 2) FE
% 3) large T inference

if nargin < 5
    indOut = ind_LP(reg);
    weight = indOut.asymV;
end

if nargin < 6
    FE = 1;
end

%% PREPARE VARIABLES
% variables in regression
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
z      = [zx zc];

% data dimensions
N       = reg.param.N;
T       = reg.param.T;

if FE == 1
    c = [c ones(N*T,1)];
    z = [z ones(N*T,1)];
end

% variable dimension
H       = size(LHS,2);
K       = size(x,2);
P       = size(c,2);
L       = size(z,2);

% Reshape variables, for later usage
zp = reshape(z',L,T,N); % L by T by N
xp = permute(reshape(x',K,T,N),[2,1,3]); % T by K by N
cp = permute(reshape(c',P,T,N),[2,1,3]); % T by P by N
yp = reshape(LHS,T,1,N,H); % T by 1 by N by H
zx = pagemtimes(zp,xp); % L by K by N
xz = pagetranspose(zx); % K by L by N
zc = pagemtimes(zp,cp); % L by P by N
zy = pagemtimes(zp,yp); % L by 1 by N by H

%% WEIGHTING MATRIX
% 1 - IV estimator;
% 2 - 2SLS;
% 3 - inverse of asym.variance;
if strcmp(weight,'IV')
    OMEGA = repmat(eye(L,L),1,1,N,H);
elseif strcmp(weight,'2SLS')
    % Omega_i = (1/T sum_t z_{it}*z_{it}')^{-1}
    zpt = permute(zp,[2,1,3]);    % z transpose
    zz  = pagemtimes(zp,zpt)/T; % L by L by N
    OMEGA = repmat(pageinv(zz),1,1,1,H);
else
    OMEGA = pageinv(weight(1:L,1:L,:,:));
end

%% PREPARE VARIABLES FOR ESTIMATION
czo = pagemtimes(zc,"transpose",OMEGA,'none');
czozc = pagemtimes(czo,zc);  % P by P by N by H
czozy = pagemtimes(czo,zy);
czozx = pagemtimes(czo,zx);
M = OMEGA - pagemtimes(czo,'transpose',pagemldivide(czozc,czo),'none');
xzm = pagemtimes(xz,M);
xzmzx = pagemtimes(xzm,zx);
xzmzy = pagemtimes(xzm,zy);

%% ITERATION PARAMS
rho     = -1/4;
tol     = 1e-6;
nIter   = 1000;  % by default, number of iteration should be less than 100

%% DATA HOLDER
Qpath  = nan(nInit,1);
bpath  = cell(nInit,1);
gpath  = nan(N,nInit);

OBJ    = nan(1,Gmax);
Gr_EST = nan(N, Gmax);
GIRF   = cell(1, Gmax);

%% ESTIMATION
for Ghat = 1:Gmax
    %     fprintf('Computing Ghat = %d \n', Ghat)
    if Ghat == 1
        % compute GIRF
        xzmzx_tmp = sum(xzmzx,3);
        xzmzy_tmp = sum(xzmzy,3);
        girf = nan(K,1,Ghat,H);
        girf(:,:,Ghat,:) = pagemldivide(xzmzx_tmp,xzmzy_tmp);

        % evalute the objective function
        phi = pagemldivide(czozc,czozy - pagemtimes(czozx,girf));
        xb = pagemtimes(xp, girf);
        cphi = pagemtimes(cp,phi);
        e_tmp = yp - xb - cphi;
        ze_tmp = pagemtimes(zp,e_tmp);
        tmp = pagemtimes(pagemtimes(ze_tmp,"transpose",OMEGA,'none'),ze_tmp);

        % store values
        Q = sum(tmp(:))/(T^2)/N;
        gr_est = ones(N,1);
    else % main algorithm starts here
        for iInit = 1:nInit
            %% STEP 1: Initialization
            ir_old = bInit(1:K,:,randsample(N,Ghat),:);
            phi_old = bInit(K+1:K+P,:,:,:);

            % initialize data holder
            ir_new  = nan(K, 1, Ghat, H);
            phi_new = nan(P, 1, N, H);
            Q_old = 999;

            for iIter = 1:nIter % iterate over assignment and updating

                %% STEP 2: Assignment
                di = nan(N, Ghat); % store the Euclidean distance b/w y and Xb for each b
                cphi = pagemtimes(cp,phi_old);
                for i = 1:Ghat
                    xb = pagemtimes(xp,ir_old(:,:,i,:));
                    e = yp - xb -cphi;
                    ze = pagemtimes(zp/T,e);
                    tmp = pagemtimes(pagemtimes(ze,'transpose',OMEGA,'none'),ze);
                    di(:,i) = sum(reshape(tmp,N,H),2);
                end
                [~,Gr] = min(di,[],2);

                uniquek = unique(Gr);
                if length(uniquek) < Ghat  % empty group
                    lucky = randperm(N);
                    for l = setdiff(1:Ghat, uniquek)
                        Gr(lucky(l)) = l;
                    end
                end

                %% STEP 3: Update Coefficient
                obj = zeros(Ghat,1);
                for g = 1:Ghat
                    yp_tmp = yp(:,:,Gr==g,:);
                    xp_tmp = xp(:,:,Gr==g);
                    cp_tmp = cp(:,:,Gr==g);
                    czozc_tmp = czozc(:,:,Gr==g,:);
                    czozy_tmp = czozy(:,:,Gr==g,:);
                    czozx_tmp = czozx(:,:,Gr==g,:);
                    xzmzx_tmp = sum(xzmzx(:,:,Gr==g,:),3);
                    xzmzy_tmp = sum(xzmzy(:,:,Gr==g,:),3);

                    ir_new(:,:,g,:) = pagemldivide(xzmzx_tmp,xzmzy_tmp);
                    phi_new(:,:,Gr==g,:) = pagemldivide(czozc_tmp,czozy_tmp - pagemtimes(czozx_tmp,ir_new(:,:,g,:)));

                    xb = pagemtimes(xp_tmp, ir_new(:,:,g,:));
                    cphi = pagemtimes(cp_tmp,phi_new(:,:,Gr==g,:));
                    e_tmp = yp_tmp - xb - cphi;
                    ze_tmp = pagemtimes(zp(:,:,Gr==g),e_tmp);
                    tmp = pagemtimes(pagemtimes(ze_tmp,'transpose',OMEGA(:,:,Gr==g,:),'none'),ze_tmp);
                    obj(g) = sum(tmp(:))/(T^2); % divide by N at the end

                end

                Q_new = sum(obj)/N;

                %% STEP 3: Convergence
                [~, converge] = resid(ir_old,ir_new,Q_old,Q_new,tol);

                if converge == 1
                    break;
                end

                Q_old  = Q_new;
                ir_old = ir_new;
                phi_old = phi_new;
            end
            Qpath(iInit) = Q_old;
            bpath{iInit} = ir_old;
            gpath(:,iInit) = Gr;
        end

        %% Final Step: find golbal minimizer across initializations
        [Q,Qid] = min(Qpath);
        gr_est  = gpath(:,Qid);
        girf    = bpath{Qid};
    end

    %% store values
    OBJ(Ghat)       = Q;
    GIRF{Ghat}      = girf;
    Gr_EST(:, Ghat) = gr_est;
end

IC = OBJ + OBJ(Gmax)*(1:Gmax)*H*(N*T)^rho;
end