function [Gr_EST, GIRF, GSE, GSE_FT] = GLP_SIM_KnownG0_Inference(reg, G0, bxInit, bcInit, weight, FE)
% GLP algorithm used for Supplemental Material S1.4
% It is the same as GLP_SIM_KnownG0.m, except that we compute both Large T and Fixed T standard errors to speed up
% By default we use mixed weighting scheme, line 211-226 (See Section S1.3)

% Related functions:
% GLP_SIM_KnownG0.m: general GLP with known G0
% GLP_SIM_UnknownG0.m: this function loops over Ghat=1,...Gmax and select Ghat by IC; since it runs over Ghat neq G0, it uses IND_LP as initial guess
% GLP_SIM_Infeasible.m: this function runs infeasible GLP (with fully pooled panel LP-IV for each group)
% GLP_SIM_Infeasible1.m: this function runs GLP_SIM_KnownG0 with 2SLS weights & true group assignment
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
%   G0: (true) number of groups
%   bxInit: initial value for beta_x (here we use the true IRs)
%   bcInit: initial value for beta_c (here we use the coef estimates from IND_LP)
%   weight: either string ('2SLS', 'IV') or user-supplied weights;
%           default: inverse of the covariance matrix of moment conditions (averaged over h)
%                   see Sec S1.3 for details
%   FE: 1 - fixed effects (within estimator, demean)

% --------------------------- OUTPUT --------------------------------
%   Gr_EST: Group composition, N by 1 vector
%   GIRF: Group IRF, K by H by G0 matrix
%   GSE: Group IRF, K by H by G0 matrix

% when weight, FE, are not supplied
% by default we use:
% 1) inverse of the covariance matrix of moment conditions (averaged over h)
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
zpt = pagetranspose(zp);
xp = permute(reshape(x',K,T,N),[2,1,3]); % T by K by N
cp = permute(reshape(c',P,T,N),[2,1,3]); % T by P by N
yp = reshape(LHS,T,1,N,H); % T by 1 by N by H
zx = pagemtimes(zp,xp); % L by K by N
xz = pagetranspose(zx); % K by L by N
zc = pagemtimes(zp,cp); % L by P by N
cz = permute(zc,[2,1,3]);
zy = pagemtimes(zp,yp); % L by 1 by N by H

%% WEIGHTING MATRIX
% 1 - IV estimator;
% 2 - 2SLS;
% 3 - inverse of asym.variance;
if strcmp(weight,'IV')
    OMEGA = repmat(eye(L,L),1,1,N,H);
elseif strcmp(weight,'2SLS')
    % Omega_i = (1/T sum_t z_{it}*z_{it}')^{-1}
    zz  = pagemtimes(zp,zpt)/T; % L by L by N
    OMEGA = repmat(pageinv(zz),1,1,1,H);
else
    OMEGA = pageinv(weight(1:L,1:L,:,:));
end

%% PREPARE VARIABLES FOR ESTIMATION
czo = pagemtimes(cz,OMEGA);
czozc = pagemtimes(czo,zc);  % P by P by N by H
czozy = pagemtimes(czo,zy);
czozx = pagemtimes(czo,zx);
M = OMEGA - pagemtimes(czo,'transpose',pagemldivide(czozc,czo),'none');
xzm = pagemtimes(xz,M);
xzmzx = pagemtimes(xzm,zx);
xzmzy = pagemtimes(xzm,zy);

%% ITERATION PARAMS
tol     = 1e-6;
nIter   = 1000;  % by default, number of iteration should be less than 100

%% ESTIMATION
if G0 == 1   % Trivial case, single group

    xzmzx_tmp = sum(xzmzx,3);
    xzmzy_tmp = sum(xzmzy,3);
    GIRF = pagemldivide(xzmzx_tmp,xzmzy_tmp);
    Gr_EST = ones(N,1);
else % main algorithm starts here
    ir_old  = bxInit;
    phi_old = bcInit(1:P,:,:,:);
    ir_new  = nan(K, 1, G0, H);
    phi_new = nan(P, 1, N, H);
    Q_old = 999;

    for iIter = 1:nIter % iterate over assignment and updating

        %% STEP 1: Assignment
        di = nan(N, G0); % store the Euclidean distance b/w y and Xb for each b
        cphi = pagemtimes(cp,phi_old);
        for i = 1:G0
            xb = pagemtimes(xp,ir_old(:,:,i,:));
            e = yp -xb -cphi;
            ze = pagemtimes(zp,e);
            tmp = pagemtimes(pagemtimes(ze,'transpose',OMEGA,'none'),ze);
            di(:,i) = sum(reshape(tmp,N,H)/(T^2),2);
        end
        [~,Gr] = min(di,[],2);

        uniquek = unique(Gr);
        if length(uniquek) < G0  % empty group
            lucky = randperm(N);
            for l = setdiff(1:G0, uniquek)
                Gr(lucky(l)) = l;
            end
        end

        %% STEP 2: Update Coefficient
        obj = zeros(G0,1);
        for g = 1:G0
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

    %% in simulation with known group number and true IRs
    %% we do not try different initializations
    Gr_EST = Gr;
    GIRF = ir_new;

end


%% Inference: Large T
GSE  = nan(K,1,G0,H);
for g = 1:G0
    yp_tmp = yp(:,:,Gr_EST==g,:);
    xp_tmp = xp(:,:,Gr_EST==g);
    cp_tmp = cp(:,:,Gr_EST==g);
    czozc_tmp = czozc(:,:,Gr_EST==g,:);
    czozy_tmp = czozy(:,:,Gr_EST==g,:);
    czozx_tmp = czozx(:,:,Gr_EST==g,:);
    xzm_tmp   = xzm(:,:,Gr_EST==g,:);
    xzmzx_tmp = sum(xzmzx(:,:,Gr_EST==g,:),3);
    phi = pagemldivide(czozc_tmp,czozy_tmp - pagemtimes(czozx_tmp,GIRF(:,:,g,:)));

    xb = pagemtimes(xp_tmp, GIRF(:,:,g,:));
    cphi = pagemtimes(cp_tmp,phi);
    e_tmp = yp_tmp - xb - cphi;

    % Sigma_g
    Sigma_g = xzmzx_tmp;
    % V_i,h
    ze_tmp = zpt(:,:,Gr_EST==g).*e_tmp;
    V_ih   = HAC4d(ze_tmp,H+1);
    % Psi_g
    Psi_g = sum(pagemtimes(pagemtimes(xzm_tmp,V_ih),'none',xzm_tmp,'transpose'),3);

    V = pagemrdivide(pagemldivide(Sigma_g,Psi_g),Sigma_g);
    GSE(:,:,g,:) = sqrt(V);
end

%% Inference: Fixed T
% STEP 1: ESTIMATE V & GAMMA (DIAGONAL)
V = zeros(K*G0*H);
Gam_tmp = zeros(K,K,H*G0);
V_j = zeros(K*H);
ytil = nan(size(yp));
for g = 1:G0
    yp_tmp = yp(:,:,Gr_EST==g,:);
    xp_tmp = xp(:,:,Gr_EST==g);
    cp_tmp = cp(:,:,Gr_EST==g);
    czozc_tmp = czozc(:,:,Gr_EST==g,:);
    czozy_tmp = czozy(:,:,Gr_EST==g,:);
    czozx_tmp = czozx(:,:,Gr_EST==g,:);
    xzm_tmp = xzm(:,:,Gr_EST==g,:);
    xzmzx_tmp = sum(xzmzx(:,:,Gr_EST==g,:),3);
    phi = pagemldivide(czozc_tmp,czozy_tmp - pagemtimes(czozx_tmp,GIRF(:,:,g,:)));

    xb = pagemtimes(xp_tmp, GIRF(:,:,g,:));
    cphi = pagemtimes(cp_tmp,phi);
    ytil(:,:,Gr_EST==g,:) = yp_tmp - cphi;
    e_tmp = ytil(:,:,Gr_EST==g,:) - xb;
    
    % Gamma
    Gam_tmp(:,:,H*(g-1)+1:H*g) = reshape(xzmzx_tmp,K,K,H);
    % V
    ze_tmp = zpt(:,:,Gr_EST==g).*e_tmp;
    for l = 1:H
        zeez_tmp = pagemtimes(ze_tmp(:,:,:,l),'transpose',ze_tmp(:,:,:,l:end),'none');
        Psi_tmp = sum(pagemtimes(pagemtimes(xzm_tmp(:,:,:,l),zeez_tmp),'none',xzm_tmp(:,:,:,l:end),'transpose'),3);
        V_j(K*(l-1)+1:K*l,K*(l-1)+1:end)=reshape(Psi_tmp,K,K*(H-l+1));
    end
    
    % V
    V(K*H*(g-1)+1:K*H*g,K*H*(g-1)+1:K*H*g) = V_j+V_j'-diag(diag(V_j));
end
Gam = kron(eye(G0*H),ones(K,K));
Gam(Gam > 0) = Gam_tmp;

% STEP 2: Correction. See Proposition S1!!
xzozx = pagemtimes(zx,'transpose',pagemtimes(OMEGA,zx),'none');
zytil = pagemtimes(zp,ytil); % L by 1 by N by H
xzozytil = pagemtimes(zx,'transpose',pagemtimes(OMEGA,zytil),'none');
for j = 1:G0
    % xzrmr(zy-zx*b_jl) K by 1 by N by H
    part_jl = xzmzy - pagemtimes(xzmzx,GIRF(:,:,j,:));
    for k = 1:G0
        if j==k
            f_ijg = zeros(1,G0,N);
            v_jg_m = zeros(K,G0,N,H);
            for g = setdiff([1:G0],j)
                IR_dif= GIRF(:,:,g,:)-GIRF(:,:,j,:);
                dist = reshape(sum(pagemtimes(IR_dif,'transpose',2*xzmzy - pagemtimes(xzmzx,GIRF(:,:,j,:)+GIRF(:,:,g,:)),'none'),4),N,1);
                bw   = std(dist)*1.06*N^(-0.2);
                f_ijg(1,g,:)  = normpdf(dist/bw)/bw .* (Gr_EST == j | Gr_EST ==g);

                % velocity: denominator 1 by 1 by N by H sum over h
                denom = sum(pagemtimes(pagemtimes(IR_dif,'transpose',xzozx,'none'),IR_dif),4);
                % velocity: scale 1 by 1 by N
                Xbi = reshape(permute(pagemtimes(xp,IR_dif),[1,4,3,2]),T*H,[]);
                scale = reshape(vecnorm(Xbi)',1,1,N);
                % velocity: numerator
                nom = xzozytil - pagemtimes(xzozx,GIRF(:,:,j,:)); % K by 1 by N by H
                v_jg_m(:,g,:,:) = nom.*scale./denom;
            end
            % K by K by N by H
            tmp = nan(K*H,K*H);
            for h = 1:H
                cor = -pagemtimes(part_jl(:,:,:,h),'none',sum(f_ijg.*v_jg_m,2),'transpose');
                tmp(K*(h-1)+1:K*h,:) = reshape(mean(cor,3),K,K*H);
            end
        else
            IR_dif= GIRF(:,:,k,:)-GIRF(:,:,j,:);
            dist = reshape(sum(pagemtimes(IR_dif,'transpose',2*xzmzy - pagemtimes(xzmzx,GIRF(:,:,j,:)+GIRF(:,:,k,:)),'none'),4),N,1);
            bw   = std(dist)*1.06*N^(-0.2);
            f_ijk = nan(1,1,N);
            f_ijk(1,1,:)  = normpdf(dist/bw)/bw .* (Gr_EST == j | Gr_EST ==g);

            % velocity: denominator 1 by 1 by N by H sum over h
            denom = sum(pagemtimes(pagemtimes(IR_dif,'transpose',xzozx,'none'),IR_dif),4);
            % velocity: scale 1 by 1 by N
            Xbi = reshape(permute(pagemtimes(xp,IR_dif),[1,4,3,2]),T*H,[]);
            scale = reshape(vecnorm(Xbi)',1,1,N);
            % velocity: numerator
            nom = xzozytil - pagemtimes(xzozx,GIRF(:,:,j,:)); % K by 1 by N by H
            v_jk_m = nom.*scale./denom;

            % K by K by N by H
            tmp = nan(K*H,K*H);
            for h = 1:H
                cor = -pagemtimes(part_jl(:,:,:,h),'none',f_ijk.*v_jk_m,'transpose');
                tmp(K*(h-1)+1:K*h,:) = reshape(mean(cor,3),K,K*H);
            end
        end
        % now we have KH by KH block of cor (tmp)
        Gam(K*H*(j-1)+1:K*H*j,K*H*(k-1)+1:K*H*k) = ...
            Gam(K*H*(j-1)+1:K*H*j,K*H*(k-1)+1:K*H*k) + tmp;
    end
end

GSE_FT = permute(reshape(diag(sqrt(Gam\V/Gam)),K,H,G0),[1,4,3,2]);
end