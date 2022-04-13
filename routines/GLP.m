function [Gr_EST, GIRF, GSE, OBJ, IC] = GLP(reg, Gmax, nInit, bInit, weight, FE, inference)
% GLP algorithm for general usage

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
%   inference: 1 - large T (IV); 2 - fixed T (IV); 3 - large T (Raw)

% --------------------------- OUTPUT --------------------------------
%   Gr_EST: Group composition, N by Gmax matrix
%   GIRF: Group IRF, 1 by Gmax cell, with K by 1 by G by H coefs 
%   GSE: Group standard errors, 1 by Gmax cell, with K by 1 by G by H SE 
%   OBJ: minimized objective function for each Ghat, 1 by Gmax vector
%   IC: Group IRF, K by H by Gmax matrix


% when weight, FE, large T are not supplied
% by default we use:
% 1) inverse of asym.variance as weighting matrix
% 2) FE
% 3) large T (Iterative)

if nargin < 5
    indOut = ind_LP(reg);
    weight = indOut.asymV;
end

if nargin < 6
    FE = 1;
end

if nargin < 7
    inference = 1; % modified large T with IV weighting
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
H       = size(LHS,2);
K       = size(x,2);
P       = size(c,2);
L       = size(z,2);

% estimation params
tol     = 1e-6;
nIter   = 1000;  % by default, number of iteration should be less than 100

if FE == 1
    % demean variables for FE estimation
    tmp    = [LHS x c z];
    tmp_de = gdemean(tmp,N,T);
    LHS    = tmp_de(:,1:H);
    x      = tmp_de(:,H+1:H+K);
    c      = tmp_de(:,H+K+1:H+K+P);
    z      = tmp_de(:,H+K+P+1:end);
end

% Reshape variables, for later usage
zp = reshape(z',L,T,N); % L by T by N
zpt = pagetranspose(zp);
xp = permute(reshape(x',K,T,N),[2,1,3]); % T by K by N
cp = permute(reshape(c',P,T,N),[2,1,3]); % T by P by N
yp = reshape(LHS,T,1,N,H); % T by 1 by N by H
zx = pagemtimes(zp,xp); % L by K by N
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
R = nan(size(OMEGA));
for i = 1:N
    for h = 1:H
        R(:,:,i,h) = chol(OMEGA(:,:,i,h));
    end
end
rzc = pagemtimes(R,zc);
czr = permute(rzc,[2,1,3,4]);
czo = pagemtimes(cz,OMEGA);
rzx = pagemtimes(R,zx);
xzr = permute(rzx,[2,1,3,4]);
rzy = pagemtimes(R,zy);

czozc = pagemtimes(czo,zc);  % P by P by N by H
czozy = pagemtimes(czo,zy);
czozx = pagemtimes(czo,zx);
M = eye(L) - pagemtimes(rzc,pagemldivide(czozc,czr));
xzrmrzx = pagemtimes(pagemtimes(xzr,M),rzx);
xzrmrzy = pagemtimes(pagemtimes(xzr,M),rzy);


%% DATA HOLDER
Qpath  = nan(nInit,1);
bpath  = cell(nInit,1);
gpath  = nan(N,nInit);

OBJ    = nan(1,Gmax);
Gr_EST = nan(N, Gmax);
GIRF   = cell(1, Gmax);
GSE    = cell(1, Gmax);

%% ESTIMATION
for Ghat = 1:Gmax
    %     fprintf('Computing Ghat = %d \n', Ghat)
    if Ghat == 1
        % compute GIRF
        xzrmrzx_tmp = sum(xzrmrzx,3);
        xzrmrzy_tmp = sum(xzrmrzy,3);
        girf = nan(K,1,Ghat,H);
        girf(:,:,Ghat,:) = pagemldivide(xzrmrzx_tmp,xzrmrzy_tmp);

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
                    e = yp -xb -cphi;
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
                    xzrmrzx_tmp = sum(xzrmrzx(:,:,Gr==g,:),3);
                    xzrmrzy_tmp = sum(xzrmrzy(:,:,Gr==g,:),3);
                    
                    ir_new(:,:,g,:) = pagemldivide(xzrmrzx_tmp,xzrmrzy_tmp);
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
                [dif, converge] = resid(ir_old,ir_new,Q_old,Q_new,tol);
                % if it converges, then break the loop
                fprintf('Initialization: %d   iteration %d: Max diff %f \n', iInit, iIter, dif)
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
        
    %% INFERENCE
    if inference == 1
        %% Large T, IV weighting matrix
        gse = nan(K,1,Ghat,H);
        %     zz  = pagemtimes(zp,'none',zp,'transpose')/T; % L by L by N
        %     OMEGA1 = repmat(pageinv(zz),1,1,1,H);
        %     [U,S,~]=pagesvd(OMEGA1,'vector');
        %     R1 = sqrt(S).*pagectranspose(U);
        OMEGA1 = repmat(eye(L,L),1,1,N,H);
        R1 = OMEGA1;
        rzc1 = pagemtimes(R1,zc);
        czo1 = pagemtimes(cz,OMEGA1);
        czozc1 = pagemtimes(czo1,zc);  % P by P by N by H
        czozy1 = pagemtimes(czo1,zy);
        czozx1 = pagemtimes(czo1,zx);
        M1 = eye(L) - pagemtimes(pagemrdivide(rzc1,czozc1),'none',rzc1,'transpose');
        rzx1 = pagemtimes(R1,zx);
        xzr1 = pagetranspose(rzx1);
        rzy1 = pagemtimes(R1,zy);
    
        xzrmr1 = pagemtimes(pagemtimes(xzr1,M1),R1);
        xzrmrzx1 = pagemtimes(pagemtimes(xzr1,M1),rzx1);
        xzrmrzy1 = pagemtimes(pagemtimes(xzr1,M1),rzy1);
        
        for g = 1:Ghat
            yp_tmp = yp(:,:,gr_est==g,:);
            xp_tmp = xp(:,:,gr_est==g);
            cp_tmp = cp(:,:,gr_est==g);
            czozc_tmp = czozc1(:,:,gr_est==g,:);
            czozy_tmp = czozy1(:,:,gr_est==g,:);
            czozx_tmp = czozx1(:,:,gr_est==g,:);
            xzrmr_tmp = xzrmr1(:,:,gr_est==g,:);
            xzrmrzx_tmp = sum(xzrmrzx1(:,:,gr_est==g,:),3);
            xzrmrzy_tmp = sum(xzrmrzy1(:,:,gr_est==g,:),3);
    
            girf(:,:,g,:) = pagemldivide(xzrmrzx_tmp,xzrmrzy_tmp);
            phi = pagemldivide(czozc_tmp,czozy_tmp - pagemtimes(czozx_tmp,girf(:,:,g,:)));
    
            xb = pagemtimes(xp_tmp, girf(:,:,g,:));
            cphi = pagemtimes(cp_tmp,phi);
            e_tmp = yp_tmp - xb - cphi;
    
            % Sigma_g
            Sigma_g = xzrmrzx_tmp;
            % V_i,h
            ze_tmp = zpt(:,:,gr_est==g).*e_tmp;
%             V_ih   = HAC4d(ze_tmp,H+1);
            Ng = sum(gr_est==g);
            V_ih = nan(L,L,Ng,H);
            for i = 1:Ng
                for h = 1:H
                    V_ih(:,:,i,h) = HAC(ze_tmp(:,:,i,h),H+1);
                end
            end
            % Psi_g
            Psi_g = sum(pagemtimes(pagemtimes(xzrmr_tmp,V_ih),'none',xzrmr_tmp,'transpose'),3);
    
            V = pagemrdivide(pagemldivide(Sigma_g,Psi_g),Sigma_g);
            gse(:,:,g,:) = sqrt(V);
        end
    
    elseif inference == 2
        %% Fixed T, IV weighting matrix
        %     zz  = pagemtimes(zp,'none',zp,'transpose')/T; % L by L by N
        %     OMEGA1 = repmat(pageinv(zz),1,1,1,H);
        %     [U,S,~]=pagesvd(OMEGA1,'vector');
        %     R1 = sqrt(S).*pagectranspose(U);
        OMEGA1 = repmat(eye(L,L),1,1,N,H);
        R1 = OMEGA1;
        rzc1 = pagemtimes(R1,zc);
        czo1 = pagemtimes(cz,OMEGA1);
        czozc1 = pagemtimes(czo1,zc);  % P by P by N by H
        czozy1 = pagemtimes(czo1,zy);
        czozx1 = pagemtimes(czo1,zx);
        M1 = eye(L) - pagemtimes(pagemrdivide(rzc1,czozc1),'none',rzc1,'transpose');
        rzx1 = pagemtimes(R1,zx1);
        xzr1 = pagetranspose(rzx1);
        rzy1 = pagemtimes(R1,zy1);
    
        xzrmr1 = pagemtimes(pagemtimes(xzr1,M1),R1);
        xzrmrzx1 = pagemtimes(pagemtimes(xzr1,M1),rzx1);
        xzrmrzy1 = pagemtimes(pagemtimes(xzr1,M1),rzy1);

        V = zeros(K*Ghat*H);
        Gam_tmp = zeros(K,K,H*Ghat);
        V_j = zeros(K*H);
        ytil = nan(size(yp));
        for g = 1:Ghat
            yp_tmp = yp(:,:,gr_est==g,:);
            xp_tmp = xp(:,:,gr_est==g);
            cp_tmp = cp(:,:,gr_est==g);
            czozc_tmp = czozc1(:,:,gr_est==g,:);
            czozy_tmp = czozy1(:,:,gr_est==g,:);
            czozx_tmp = czozx1(:,:,gr_est==g,:);
            xzrmr_tmp = xzrmr1(:,:,gr_est==g,:);
            xzrmrzx_tmp = sum(xzrmrzx1(:,:,gr_est==g,:),3);
            xzrmrzy_tmp = sum(xzrmrzy1(:,:,gr_est==g,:),3);
    
            girf(:,:,g,:) = pagemldivide(xzrmrzx_tmp,xzrmrzy_tmp);
            phi = pagemldivide(czozc_tmp,czozy_tmp - pagemtimes(czozx_tmp,girf(:,:,g,:)));
    
            xb = pagemtimes(xp_tmp, girf(:,:,g,:));
            cphi = pagemtimes(cp_tmp,phi);
            ytil(:,:,gr_est==g,:) = yp_tmp - cphi;
            e_tmp = ytil(:,:,gr_est==g,:) - xb;
    
            % Gamma
            Gam_tmp(:,:,H*(g-1)+1:H*g) = reshape(xzrmrzx_tmp,K,K,H);
            % V
            ze_tmp = zpt(:,:,gr_est==g).*e_tmp;
            for l = 1:H
                zeez_tmp = pagemtimes(ze_tmp(:,:,:,l),'transpose',ze_tmp(:,:,:,l:end),'none');
                Psi_tmp = sum(pagemtimes(pagemtimes(xzrmr_tmp(:,:,:,l),zeez_tmp),'none',xzrmr_tmp(:,:,:,l:end),'transpose'),3);
                V_j(K*(l-1)+1:K*l,K*(l-1)+1:end)=reshape(Psi_tmp,K,K*(H-l+1));
            end
    
            % V
            V(K*H*(g-1)+1:K*H*g,K*H*(g-1)+1:K*H*g) = V_j+V_j'-diag(diag(V_j));
        end
        Gam = kron(eye(Ghat*H),ones(K,K));
        Gam(Gam > 0) = Gam_tmp;
    
        % STEP 2: Correction. See Proposition S1!!
        xzozx = pagemtimes(zx,'transpose',pagemtimes(OMEGA1,zx),'none');
        zytil = pagemtimes(zp,ytil); % L by 1 by N by H
        xzozytil = pagemtimes(zx,'transpose',pagemtimes(OMEGA1,zytil),'none');
        for j = 1:Ghat
            % xzrmr(zy-zx*b_jl) K by 1 by N by H
            part_jl = xzrmrzy1 - pagemtimes(xzrmrzx1,girf(:,:,j,:));
            for k = 1:Ghat
                if j==k
                    f_ijg = zeros(1,Ghat,N);
                    v_jg_m = zeros(K,Ghat,N,H);
                    for g = setdiff([1:Ghat],j)
                        IR_dif= girf(:,:,g,:)-girf(:,:,j,:);
                        dist = reshape(sum(pagemtimes(IR_dif,'transpose',2*xzrmrzy1 - pagemtimes(xzrmrzx1,girf(:,:,j,:)+girf(:,:,g,:)),'none'),4),N,1);
                        bw   = std(dist)*1.06*N^(-0.2);
                        f_ijg(1,g,:)  = normpdf(dist/bw)/bw .* (gr_est == j | gr_est ==g);
    
                        % velocity: denominator 1 by 1 by N by H sum over h
                        denom = sum(pagemtimes(pagemtimes(IR_dif,'transpose',xzozx,'none'),IR_dif),4);
                        % velocity: scale 1 by 1 by N
                        Xbi = reshape(permute(pagemtimes(xp,IR_dif),[1,4,3,2]),T*H,[]);
                        scale = reshape(vecnorm(Xbi)',1,1,N);
                        % velocity: numerator
                        nom = xzozytil - pagemtimes(xzozx,girf(:,:,j,:)); % K by 1 by N by H
                        v_jg_m(:,g,:,:) = nom.*scale./denom;
                    end
                    % K by K by N by H
                    tmp = nan(K*H,K*H);
                    for h = 1:H
                        cor = -pagemtimes(part_jl(:,:,:,h),'none',sum(f_ijg.*v_jg_m,2),'transpose');
                        tmp(K*(h-1)+1:K*h,:) = reshape(mean(cor,3),K,K*H);
                    end
                else
                    IR_dif= girf(:,:,k,:)-girf(:,:,j,:);
                    dist = reshape(sum(pagemtimes(IR_dif,'transpose',2*xzrmrzy1 - pagemtimes(xzrmrzx1,girf(:,:,j,:)+girf(:,:,k,:)),'none'),4),N,1);
                    bw   = std(dist)*1.06*N^(-0.2);
                    f_ijk = nan(1,1,N);
                    f_ijk(1,1,:)  = normpdf(dist/bw)/bw .* (gr_est == j | gr_est ==g);
    
                    % velocity: denominator 1 by 1 by N by H sum over h
                    denom = sum(pagemtimes(pagemtimes(IR_dif,'transpose',xzozx,'none'),IR_dif),4);
                    % velocity: scale 1 by 1 by N
                    Xbi = reshape(permute(pagemtimes(xp,IR_dif),[1,4,3,2]),T*H,[]);
                    scale = reshape(vecnorm(Xbi)',1,1,N);
                    % velocity: numerator
                    nom = xzozytil - pagemtimes(xzozx,girf(:,:,j,:)); % K by 1 by N by H
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
    
        gse = permute(reshape(diag(sqrt(Gam\V/Gam)),K,H,Ghat),[1,4,3,2]);
    
    else
        % Large T, raw weighting matrix
        gse = nan(K,1,Ghat,H);
        xzrmr = pagemtimes(pagemtimes(xzr,M),R);
        for g = 1:Ghat
            yp_tmp = yp(:,:,gr_est==g,:);
            xp_tmp = xp(:,:,gr_est==g);
            cp_tmp = cp(:,:,gr_est==g);
            czozc_tmp = czozc(:,:,gr_est==g,:);
            czozy_tmp = czozy(:,:,gr_est==g,:);
            czozx_tmp = czozx(:,:,gr_est==g,:);
            xzrmr_tmp = xzrmr(:,:,gr_est==g,:);
            xzrmrzx_tmp = sum(xzrmrzx(:,:,gr_est==g,:),3);
            phi = pagemldivide(czozc_tmp,czozy_tmp - pagemtimes(czozx_tmp,girf(:,:,g,:)));
    
            xb = pagemtimes(xp_tmp, girf(:,:,g,:));
            cphi = pagemtimes(cp_tmp,phi);
            e_tmp = yp_tmp - xb - cphi;
    
            % Sigma_g
            Sigma_g = xzrmrzx_tmp;
            % V_i,h
            ze_tmp = zpt(:,:,gr_est==g).*e_tmp;
            V_ih = HAC4d(ze_tmp,H+1);
            % Psi_g
            Psi_g = sum(pagemtimes(pagemtimes(xzrmr_tmp,V_ih),'none',xzrmr_tmp,'transpose'),3);
    
            V = pagemrdivide(pagemldivide(Sigma_g,Psi_g),Sigma_g);
            gse(:,:,g,:) = sqrt(V);
        end
    
    end

    %% STORE OUTPUT
    OBJ(Ghat)       = Q;    
    GIRF{Ghat}      = girf;
    GSE{Ghat}       = gse;
    Gr_EST(:, Ghat) = gr_est;

end
    

IC = OBJ + OBJ(Gmax)*[1:Gmax]*H*(N*T)^(-0.3);
end