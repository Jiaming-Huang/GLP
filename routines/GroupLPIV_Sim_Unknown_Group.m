function [Gr_EST, GIRF, OBJ, IC] = GroupLPIV_Sim_Unknown_Group(reg, Gmax, nInit, bInit, weight, FE)
% I assume heterogeneous coef of controls --> project out controls first
% The program is written for the case with scalar x and scalar z

% INPUT:
%   reg: data (reg.x, reg.y, reg.z, reg.LHS, reg.control, reg.param)
%   G0: number of groups to be classified
%   ninit: number of initializations
%   binit: potential initial values, see blow
%   weight: 1 - inverse of asym.variance; 2 - IV estimator; 3 - 2SLS;
%   FE: 1 - fixed effects (within estimator); 2 - random effects (OLS)
%   inference: 1 - large T; 2 - post GLP; 3- fixed T

% OUTPUT:
%   Group: Group composition, N by 1 vector
%   GIRF: Group IRF, K by H+1 matrix
%   Qpath: path of LOSS
%   gpath: path of groups
%   bpath: path of coef (could be super large) K by h by ninit



% when weight, FE, large T are not supplied
% by default we use:
% 1) inverse of asym.variance as weighting matrix
% 2) FE
% 3) large T inference

if nargin < 5
    weight = 0;
end

if nargin < 6
    FE = 1;
end



%% UNPACK VARIABLES & PARAMETER SETTINGS
% variables in regression
LHS = reg.LHS;
x   = reg.x;
z   = reg.z;
control = reg.control;

% parameters
H = size(LHS,2);  % it's actually H+1, see prepare_GLP.m
K = size(x,2);
P = size(control,2);

N = reg.param.N;
T = reg.param.T;

% estimation params
tol = 1e-6;
nIter = 100;  % by default, number of iteration should be less than 100

% data holder
Qpath  = nan(nInit,1);
bpath  = cell(nInit,1);
gpath  = nan(N,nInit);

IC     = nan(1,Gmax);
OBJ    = nan(1,Gmax);
Gr_EST = nan(N, Gmax);
GIRF   = cell(1, Gmax);

%% STEP 1: PARTIALING OUT CONTROLS
% if FE=1, demean
% if there are controls, partial them out unit by unit

if P == 0 % no controls
    if FE == 1
        tmp    = [LHS x];
        tmp_de = gdemean(tmp,N,T);
        y_p    = tmp_de(:,1:H);
        x_p    = tmp_de(:,H+1:H+K);
    else
        y_p = LHS;
        x_p = x;
    end
    z_p = z;
    
else
    if FE == 1
        % demean
        tmp    = [LHS x control];
        tmp_de = gdemean(tmp,N,T);
        LHS    = tmp_de(:,1:H);
        x      = tmp_de(:,H+1:H+K);
        control= tmp_de(:,H+K+1:end);
    end
    % Partial out controls ===> x_p and z_p are all NT by 1 vectors!
    for i = 1: N
        ytmp = LHS(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        ztmp = z(T*(i-1)+1:T*i,:);
        wtmp = control(T*(i-1)+1:T*i,:);
        
        % partial out controls
        y_p(T*(i-1)+1:T*i,:) = ytmp - wtmp*((wtmp'*wtmp)\(wtmp'*ytmp));
        x_p(T*(i-1)+1:T*i,:) = xtmp - wtmp*((wtmp'*wtmp)\(wtmp'*xtmp));
        z_p(T*(i-1)+1:T*i,:) = ztmp - wtmp*((wtmp'*wtmp)\(wtmp'*ztmp));
    end
    
end

%% STEP 1.5: WEIGHTING MATRIX
% 1 - inverse of asym.variance;
% 2 - IV estimator;
% 3 - 2SLS;
if size(weight,1) ~= 1
    %     [~, se, ~] = ind_LP(reg);
    %     vbeta = se.^2;  % N by H
    vbeta = weight.^2;  % speed up, no need to estimate ind_LP many times
    OMEGA = reshape(1./vbeta,1, N*H);
elseif weight == 2
    OMEGA = ones(1,N*H);
else
    tmp = 1 ./ mean(reshape(z_p.^2,T,N));
    OMEGA = repmat(tmp,1,H);
end



%% ESTIMATION STARTS HERE
% CASE 1: SINGLE GROUP
%           it is as if panel LP-IV
%           but to keep objective functions comparable across Gguess
%           I still use horizon-specific weighting matrix
% CASE 2: G_guess GROUP
%           See the main algorithm in the paper




for G0 = 1:Gmax
%     fprintf('Computing G0 = %d \n', G0)
    if G0 == 1
        zx    = mean(reshape(z_p.*x_p,T,N)); % 1 by N
        Omega = reshape(OMEGA,N,H);          % N by H
        zxoxz = zx*(zx'.*Omega);             % 1 by H
        zy    = reshape(mean(reshape(z_p.*y_p,T,N*H)),N,H); % N by H
        zxozy = zx*(Omega.*zy);
        b = zxozy./zxoxz;
        
        % get the OBJ
        mbar = mean(reshape(z_p.*(y_p - x_p*b),T,N*H)); % 1 by NH
        Q = sum(mbar.^2.*OMEGA)/N;
        
        % assign outputs
        gr_est = ones(N,1);
        girf = b;
        
        
    else
        for iInit = 1:nInit
            %% STEP 2: INITIALIZATION
            idx = randperm(N);
            
            if isempty(bInit)
                % find global optimum in randomly chosen G0+3 units
                % you can change G0+3 to anything larger than G0
                % BUT KEEP IT SMALL O.W. IT TAKES AGES
                part = partitions(G0+3,G0);
                obj_init = zeros(length(part),1);
                binit_partition = cell(length(part),1);
                
                for j = 1:length(part) % for each possible partition
                    obj = zeros(G0,1);
                    bInit = nan(G0,H);
                    for l = 1:G0
                        g = part{j,1}{1,l};  % e.g. g = [4 5]
                        ind = zeros(N,1); ind(idx(g))=1; ind = logical(kron(ind,ones(T,1)));
                        Omega_tmp = Omega(idx(g));   % Ng by 1
                        ytmp = y_p(ind,:);
                        xtmp = x_p(ind,:);
                        ztmp = z_p(ind,:);
                        xhat = ztmp*((ztmp'*ztmp)\(ztmp'*xtmp));
                        bInit(l,:) = ((xhat'*xhat)\(xhat'*ytmp));
                        m = ztmp.*(ytmp - xtmp*bInit(l,:));
                        m = reshape(m,T,size(m,1)/T,H); %assignment, doesn't matter if not scaled by T
                        obj(l) = mean(sum(Omega_tmp.*( sum(m).^2),3)); % sum across H and take the mean over N
                    end
                    binit_partition{j} = bInit;
                    obj_init(j) = sum(obj);
                end
                % select the global minimal in this subsample as our initial guess
                [~,dum] = min(obj_init);
                b_old = binit_partition{dum};
            else
                % if binit provided, randomly choose G_guess IRs as guess
                b_old = bInit(idx(1:G0),:);
            end
            
            % initialize data holder
            b_new = nan(G0,size(LHS,2));
            Q_old = 999;
            QQtmp = nan(nIter,1);
            
            for iIter = 1:nIter % iterate over assignment and updating
                
                %% STEP 3: ASSIGNMENT
                di = nan(N, G0); % store the Euclidean distance b/w y and Xb for each b
                for g = 1:G0
                    qi = mean(reshape(z_p.*(y_p - x_p*b_old(g,:)),T,N*H)).^2.*OMEGA; % 1 by NH
                    di(:,g) = sum(reshape(qi,N,H),2);  % sum over h but not i
                end
                [~,Gr] = min(di,[],2);
                
                % if there're empty groups
                uniquek = unique(Gr);
                if length(uniquek) < G0
                    lucky = randperm(N);
                    for l = setdiff(1:G0, uniquek)
                        Gr(lucky(l)) = l;
                    end
                end
                
                %% STEP 4: UPDATE
                GrLong = kron(Gr,ones(T,1)); % reshape NT by 1
                obj = zeros(G0,1);
                for g = 1:G0
                    % for each group, assign variables
                    Ng = sum(Gr==g);
                    ytmp = y_p(GrLong==g,:);
                    xtmp = x_p(GrLong==g,:);
                    ztmp = z_p(GrLong==g,:);
                    Omega1 = OMEGA(repmat((Gr==g)',1,H));  %Omega1 and Omega2 are the same, just with different shape
                    Omega2 = reshape(Omega1, Ng,H);
                    
                    % apply GMM formula as above
                    zx   = mean(reshape(ztmp.*xtmp,T,Ng));
                    zxoxz = zx*(zx'.*Omega2);
                    zy    = reshape(mean(reshape(ztmp.*ytmp,T,Ng*H)),Ng,H);
                    zxozy = zx*(zy.*Omega2);
                    b_new(g,:) = zxozy./zxoxz;
                    mbar = mean(reshape(ztmp.*(ytmp - xtmp*b_new(g,:)),T,Ng*H));
                    obj(g) = sum(mbar.^2.*Omega1); % divide by N at the end
                end
                
                Q_new = sum(obj)/N;
                
                %% STEP 5: CONVERGENCE
                % compute convergence criterion
                [~, d] = resid(b_old,b_new,Q_old,Q_new,tol);
                
                % if it converges, then break the loop
                %             fprintf('Initialization: %d   iteration %d: Max diff %f \n', iInit, iIter, dif)
                if d == 1
                    break;
                end
                
                % prepare values for the next loop
                QQtmp(iIter)=Q_new;
                Q_old = Q_new;
                b_old = b_new;
            end
            Qpath(iInit) = Q_old;
            bpath{iInit} = b_old;
            gpath(:,iInit) = Gr;
        end
        
        %% Final Step: find golbal minimizer across initializations
        [Q,Qid] = min(Qpath);
        gr_est  = gpath(:,Qid);
        girf    = bpath{Qid};
    end
    
    %% store values
    Gr_EST(:, G0) = gr_est;
    GIRF{G0} = girf;
    
    
    %% FIND GROUP NUMBER
    OBJ(G0) = Q;
end

IC = OBJ + OBJ(Gmax)*[1:Gmax]*H*(N*T)^(-0.2);
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