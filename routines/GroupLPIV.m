function [Gr_EST, GIRF, GSE, IC] = GroupLPIV(reg, Gmax, nInit, bInit, weight, FE, inference)
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

if nargin < 7
    inference = 1;
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
Qpath = nan(nInit,1);
bpath = cell(nInit,1);
gpath = nan(N,nInit);

% output
IC     = nan(1,Gmax);
OBJ    = nan(1,Gmax);
Gr_EST = nan(N, Gmax);
GIRF   = cell(1, Gmax);
GSE    = cell(1,Gmax);

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
    y_p = nan(size(LHS));
    x_p = nan(size(x));
    z_p = nan(size(z));
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
    fprintf('Computing G0 = %d \n', G0)
    % GSE is computed at the end
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
                [dif, d] = resid(b_old,b_new,Q_old,Q_new,tol);
                
                % if it converges, then break the loop
                fprintf('Initialization: %d   iteration %d: Max diff %f \n', iInit, iIter, dif)
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
    
    
    
    
    %% Inference
    gse = nan(G0,H);
    if inference == 1
        %% Large T inference
        % See Theorem 2
        GrLong = kron(gr_est,ones(T,1));
        for g = 1:G0
            % as is shown in main text, the scaling (Ng vs N) does not matter
            % assign variables
            Ng = sum(gr_est==g);
            ytmp = y_p(GrLong==g,:);
            xtmp = x_p(GrLong==g,:);
            ztmp = z_p(GrLong==g,:);
            etmp = ytmp - xtmp*girf(g,:);
            Omega1 = OMEGA(repmat((gr_est==g)',1,H));  %Omega1 and Omega2 are the same, just with different shape
            Omega2 = reshape(Omega1, Ng,H);
            % find Sigma_g (sum but not mean)
            zx   = mean(reshape(ztmp.*xtmp,T,Ng));
            SIGMA = zx*(zx'.*Omega2);  % 1 by H
            % find Phi_g
            zep2 = reshape(sum(reshape( (ztmp.*etmp).^2,T,Ng*H)),Ng,H);
            for h = 1:H
                PHI =(zx.*Omega2(:,h)').^2*zep2(:,h)/T;
                gse(g,h) = PHI/(SIGMA(h)^2)/T;
            end
        end
        
    elseif inference == 2
        %% Post-GLP inference: use panel LP with the estimated groups
        % use panel LP with the estimated groups
        gtmp = kron(gr_est,ones(T,1));
        tmp = [];
        if isempty(control)
            for g = 1:G0
                tmp.y = reg.y(gtmp==g,:);
                tmp.x = reg.x(gtmp==g,:);
                tmp.z = reg.z(gtmp==g,:);
                tmp.LHS = reg.LHS(gtmp==g,:);
                tmp.control = [];
                tmp.param.N = sum(gr_est==g);
                tmp.param.T = reg.param.T;
                [girf(g,:), gse(g,:)] = panel_LP(tmp,FE);
            end
        else
            for g = 1:G0
                tmp.y = reg.y(gtmp==g,:);
                tmp.x = reg.x(gtmp==g,:);
                tmp.z = reg.z(gtmp==g,:);
                tmp.LHS = reg.LHS(gtmp==g,:);
                tmp.control = reg.control(gtmp==g,:);
                tmp.param.N = sum(gr_est==g);
                tmp.param.T = reg.param.T;
                [girf(g,:), gse(g,:)] = panel_LP(tmp,FE);
            end
        end
        
    else
        %% Fixed T inference
        GrLong   = kron(gr_est,ones(T,1));
        V = zeros(G0*H);
        Gam = zeros(G0*H);
        for g = 1:G0
            Ng = sum(gr_est==g);
            %         NG(g,g) = Ng;
            ytmp = y_p(GrLong==g,:);
            xtmp = x_p(GrLong==g,:);
            ztmp = z_p(GrLong==g,:);
            etmp = ytmp - xtmp*girf(g,:);
            
            % compute Gam
            Omega      = reshape(OMEGA(repmat((gr_est==g)',1,H)),Ng,H);
            zx         = sum(reshape(ztmp.*xtmp,T,Ng));
            zxoxz      = zx*(zx'.*Omega);
            
            % compute V
            for h = 1:H
                Gam(G0*(h-1)+g,G0*(h-1)+g) = 2*zxoxz(h)/N;
                for h_til = h:H
                    zep      = sum(reshape((ztmp.*etmp(:,h)).*(etmp(:,h_til).*ztmp),T,Ng));
                    V(G0*(h-1)+g,G0*(h_til-1)+g) = 4*((zx.*Omega(:,h)').*(zx.*Omega(:,h_til)'))*zep'/N;
                end
            end
        end
        V = (V+V') - eye(size(V,1)).*diag(V);
        
        OMEGA1 = reshape(OMEGA,N,H); % N by H
        zxo = sum(reshape(z_p.*x_p,T,N))'.*OMEGA1;
        xzozx = sum(reshape(z_p.*x_p,T,N)).^2'.*OMEGA1;
        
        for g = 1:G0
            for gtil = 1:G0
                if g == gtil
                    xzozep = reshape(sum(reshape(z_p.*(y_p - x_p.*girf(g,:)),T,N*H)),N,H).*zxo;
                    for h = 1:H
                        % term A, term D depends on h and htil
                        % term B, term C and den depends on l
                        termA = xzozep(:,h);
                        termD = xzozep;
                        %termA = xzozep(:,h);
                        %termD = xzozep(:,h:end);
                        % denominator
                        termB = zeros(N,G0);
                        termC = zeros(N,G0);
                        % density
                        den = zeros(N,G0);
                        for l = 1:G0
                            if l ~= g
                                IRd = girf(l,:)-girf(g,:);
                                termB(:,l) = sum(xzozx.*(IRd.^2),2);
                                tmpC = reshape((x_p.*IRd).^2,T,N,H);
                                tmpC1 = reshape((x_p.*IRd),T,N,H);
                                termC(:,l) = (sum(reshape(permute(tmpC,[1 3 2]),T*H,N))./vecnorm(reshape(permute(tmpC1,[1 3 2]),T*H,N)))';
                                
                                % compute density
                                zepl = reshape(sum(reshape(z_p.*(y_p-x_p.*girf(l,:)),T,N*H)),N,H);
                                zepg = reshape(sum(reshape(z_p.*(y_p-x_p.*girf(g,:)),T,N*H)),N,H);
                                dist = sum(zepl.*OMEGA1.*zepl - zepg.*OMEGA1.*zepg,2);
                                bw   = std(dist)*1.06*N^(-0.2);
                                den(:,l)  = normpdf(dist/bw)/bw .* (gr_est == l | gr_est ==g);
                            end
                        end
                        cor = -2*mean(nansum(den.*termC./termB,2).*termA.*termD);
                        for htil = h:H
                            Gam(G0*(h-1)+g,G0*(htil-1)+g) = Gam(G0*(h-1)+g,G0*(htil-1)+g) + cor(htil);
                        end
                    end
                else
                    xzozep_g = reshape(sum(reshape(z_p.*(y_p - x_p.*girf(g,:)),T,N*H)),N,H).*zxo;
                    xzozep_gtil = reshape(sum(reshape(z_p.*(y_p - x_p.*girf(gtil,:)),T,N*H)),N,H).*zxo;
                    for h = 1:H
                        termA = xzozep_g(:,h);
                        termD = xzozep_gtil;
                        
                        IRd = girf(gtil,:)-girf(g,:);
                        termB = sum(xzozx.*(IRd.^2),2);
                        tmpC = reshape((x_p.*IRd).^2,T,N,H);
                        tmpC1 = reshape((x_p.*IRd),T,N,H);
                        termC = (sum(reshape(permute(tmpC,[1 3 2]),T*H,N))./vecnorm(reshape(permute(tmpC1,[1 3 2]),T*H,N)))';
                        
                        % compute density
                        zepgtil = reshape(sum(reshape(z_p.*(y_p-x_p.*girf(gtil,:)),T,N*H)),N,H);
                        zepg = reshape(sum(reshape(z_p.*(y_p-x_p.*girf(g,:)),T,N*H)),N,H);
                        dist  = sum(zepgtil.*OMEGA1.*zepgtil - zepg.*OMEGA1.*zepg,2);
                        bw    = std(dist)*1.06*N^(-0.2);
                        den   = normpdf(dist/bw)/bw .* (gr_est == gtil | gr_est ==g);
                        cor = 2*mean(termA.*termC.*den./termB .*termD);
                        for htil = h:H
                            Gam(G0*(h-1)+g,G0*(htil-1)+gtil) = Gam(G0*(h-1)+g,G0*(htil-1)+gtil) + cor(htil);
                        end
                    end
                    
                end
            end
        end
        
        
        Gam = (triu(Gam)+triu(Gam)') - eye(size(Gam,1)).*diag(Gam);
        gse = reshape(diag(sqrt(Gam\V/Gam/N)),G0,H);
        
    end
    
    %% store values
    Gr_EST(:, G0) = gr_est;
    GIRF{G0} = girf;
    GSE{G0} = gse;
    
    %% FIND GROUP NUMBER - IC
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