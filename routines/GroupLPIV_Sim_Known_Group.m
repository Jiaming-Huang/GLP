function [Gr_EST, GIRF, GSE] = GroupLPIV_Sim_Known_Group(reg, G0, bTrue, weight, FE, inference)
% GLP that initialize with true IRs
% This is used for simulation only - run one iteration to speed up

% INPUT:
%   reg: data (reg.x, reg.y, reg.z, reg.LHS, reg.control, reg.param)
%   Gguess: number of groups to be classified
%   bTrue: the true IRs, used as initial value
%   weight: 1 - inverse of asym.variance; 2 - IV estimator; 3 - 2SLS;
%   FE: 1 - fixed effects (within estimator); 2 - random effects (OLS)
%   inference: 1 - large T; 2 - post GLP; 3- fixed T

% OUTPUT:
%   Group: Group composition, N by 1 vector
%   GIRF: Group IRF, K by H+1 matrix


% when weight, FE, large T are not supplied
% by default we use:
% 1) inverse of asym.variance as weighting matrix
% 2) FE
% 3) large T inference

if nargin < 4
    weight = 0;
end

if nargin < 5
    FE = 1;
end

if nargin < 6
    inference = 1;
end

%% UNPACK VARIABLES
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



%% WEIGHTING MATRIX
% 1 - inverse of asym.variance;
% 2 - IV estimator;
% 3 - 2SLS;
if size(weight,1) ~= 1
    %     [~, se, ~] = ind_LP(reg);
    %     vbeta = se.^2;  % N by H
    vbeta = weight.^2;
    OMEGA = reshape(1./vbeta,1, N*H);
elseif weight == 2
    OMEGA = ones(1,N*H);
else
    tmp = 1 ./ mean(reshape(z_p.^2,T,N));
    OMEGA = repmat(tmp,1,H);
end


%% ESTIMATION
% See section A.2 for analytical formula


if G0 == 1   % Trivial case, single group
    zx    = mean(reshape(z_p.*x_p,T,N)); % 1 by N
    Omega = reshape(OMEGA,N,H);          % N by H
    zxoxz = zx*(zx'.*Omega);             % 1 by H
    zy    = reshape(mean(reshape(z_p.*y_p,T,N*H)),N,H); % N by H
    zxozy = zx*(Omega.*zy);
    b = zxozy./zxoxz;
    % get the OBJ
    %mbar = mean(reshape(z_p.*(y_p - x_p*b),T,N*H)); % 1 by NH
    %Qpath = sum(mbar.^2.*OMEGA)/N;
    Gr_EST = ones(N,1);
    GIRF = b;
    
else % main algorithm starts here
    b_old = bTrue;
    b_new = nan(G0,size(y_p,2));
    Q_old = 999;
    
    for iter = 1:nIter % iterate over assignment and updating
        
        %% STEP 3: Assignment
        di = nan(N, G0); % store the Euclidean distance b/w y and Xb for each b
        for i = 1:G0
            qi = mean(reshape(z_p.*(y_p - x_p*b_old(i,:)),T,N*H)).^2.*OMEGA; % 1 by NH
            di(:,i) = sum(reshape(qi,N,H),2);  % sum over h but not i
        end
        [~,Gr] = min(di,[],2);
        
        uniquek = unique(Gr);
        if length(uniquek) < G0  % empty group
            lucky = randperm(N);
            for l = setdiff(1:G0, uniquek)
                Gr(lucky(l)) = l;
            end
        end
        
        %% STEP 4: Update
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
        
        %% STEP 5: Convergence
        [~, converge] = resid(b_old,b_new,Q_old,Q_new,tol);
        
        if converge == 1
            break;
        end
        
        Q_old = Q_new;
        b_old = b_new;
    end
    
    %% in simulation with known group number and true IRs
    %% we do not try different initializations
    Gr_EST = Gr;
    GIRF = b_old;
    
end




%% Inference
GSE = nan(G0,H);

if inference == 1
    %% Large T inference
    % See Theorem 2
    GrLong = kron(Gr_EST,ones(T,1));
    for g = 1:G0
        % as is shown in main text, the scaling (Ng vs N) does not matter
        % assign variables
        Ng = sum(Gr_EST==g);
        ytmp = y_p(GrLong==g,:);
        xtmp = x_p(GrLong==g,:);
        ztmp = z_p(GrLong==g,:);
        etmp = ytmp - xtmp*GIRF(g,:);
        Omega1 = OMEGA(repmat((Gr_EST==g)',1,H));  %Omega1 and Omega2 are the same, just with different shape
        Omega2 = reshape(Omega1, Ng,H);
        % find Sigma_g (sum but not mean)
        zx   = mean(reshape(ztmp.*xtmp,T,Ng));
        SIGMA = zx*(zx'.*Omega2);  % 1 by H
        % find Phi_g
        zep2 = reshape(sum(reshape( (ztmp.*etmp).^2,T,Ng*H)),Ng,H);
        for h = 1:H
            PHI =(zx.*Omega2(:,h)').^2*zep2(:,h)/T;
            GSE(g,h) = PHI/(SIGMA(h)^2)/T;
        end
    end
    
elseif inference == 2
    % use panel LP with the estimated groups
    gtmp = kron(Gr_EST,ones(T,1));
    tmp = [];
    if isempty(control)
        for g = 1:G0
            tmp.y = reg.y(gtmp==g,:);
            tmp.x = reg.x(gtmp==g,:);
            tmp.z = reg.z(gtmp==g,:);
            tmp.LHS = reg.LHS(gtmp==g,:);
            tmp.control = [];
            tmp.param.N = sum(Gr_EST==g);
            tmp.param.T = reg.param.T;
            [GIRF(g,:), GSE(g,:)] = panel_LP(tmp,FE);
        end
    else
        for g = 1:G0
            tmp.y = reg.y(gtmp==g,:);
            tmp.x = reg.x(gtmp==g,:);
            tmp.z = reg.z(gtmp==g,:);
            tmp.LHS = reg.LHS(gtmp==g,:);
            tmp.control = reg.control(gtmp==g,:);
            tmp.param.N = sum(Gr_EST==g);
            tmp.param.T = reg.param.T;
            [GIRF(g,:), GSE(g,:)] = panel_LP(tmp,FE);
        end
    end
elseif inference == 3
    GrLong   = kron(Gr_EST,ones(T,1));
    V = zeros(G0*H);
    Gam = zeros(G0*H);
    for g = 1:G0
        Ng = sum(Gr_EST==g);
        %         NG(g,g) = Ng;
        ytmp = y_p(GrLong==g,:);
        xtmp = x_p(GrLong==g,:);
        ztmp = z_p(GrLong==g,:);
        etmp = ytmp - xtmp*GIRF(g,:);
        
        % compute Gam
        Omega      = reshape(OMEGA(repmat((Gr_EST==g)',1,H)),Ng,H);
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
                xzozep = reshape(sum(reshape(z_p.*(y_p - x_p.*GIRF(g,:)),T,N*H)),N,H).*zxo;
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
                            IRd = GIRF(l,:)-GIRF(g,:);
                            termB(:,l) = sum(xzozx.*(IRd.^2),2);
                            tmpC = reshape((x_p.*IRd).^2,T,N,H);
                            tmpC1 = reshape((x_p.*IRd),T,N,H);
                            termC(:,l) = (sum(reshape(permute(tmpC,[1 3 2]),T*H,N))./vecnorm(reshape(permute(tmpC1,[1 3 2]),T*H,N)))';
                            
                            % compute density
                            zepl = reshape(sum(reshape(z_p.*(y_p-x_p.*GIRF(l,:)),T,N*H)),N,H);
                            zepg = reshape(sum(reshape(z_p.*(y_p-x_p.*GIRF(g,:)),T,N*H)),N,H);
                            dist = sum(zepl.*OMEGA1.*zepl - zepg.*OMEGA1.*zepg,2);
                            bw   = std(dist)*1.06*N^(-0.2);
                            den(:,l)  = normpdf(dist/bw)/bw .* (Gr_EST == l | Gr_EST ==g);
                        end
                    end
                    cor = -2*mean(nansum(den.*termC./termB,2).*termA.*termD);
                    for htil = h:H
                        Gam(G0*(h-1)+g,G0*(htil-1)+g) = Gam(G0*(h-1)+g,G0*(htil-1)+g) + cor(htil);
                    end
                end
            else
                xzozep_g = reshape(sum(reshape(z_p.*(y_p - x_p.*GIRF(g,:)),T,N*H)),N,H).*zxo;
                xzozep_gtil = reshape(sum(reshape(z_p.*(y_p - x_p.*GIRF(gtil,:)),T,N*H)),N,H).*zxo;
                for h = 1:H
                    termA = xzozep_g(:,h);
                    termD = xzozep_gtil;
                    
                    IRd = GIRF(gtil,:)-GIRF(g,:);
                    termB = sum(xzozx.*(IRd.^2),2);
                    tmpC = reshape((x_p.*IRd).^2,T,N,H);
                    tmpC1 = reshape((x_p.*IRd),T,N,H);
                    termC = (sum(reshape(permute(tmpC,[1 3 2]),T*H,N))./vecnorm(reshape(permute(tmpC1,[1 3 2]),T*H,N)))';
                    
                    % compute density
                    zepgtil = reshape(sum(reshape(z_p.*(y_p-x_p.*GIRF(gtil,:)),T,N*H)),N,H);
                    zepg = reshape(sum(reshape(z_p.*(y_p-x_p.*GIRF(g,:)),T,N*H)),N,H);
                    dist  = sum(zepgtil.*OMEGA1.*zepgtil - zepg.*OMEGA1.*zepg,2);
                    bw    = std(dist)*1.06*N^(-0.2);
                    den   = normpdf(dist/bw)/bw .* (Gr_EST == gtil | Gr_EST ==g);
                    cor = 2*mean(termA.*termC.*den./termB .*termD);
                    for htil = h:H
                        Gam(G0*(h-1)+g,G0*(htil-1)+gtil) = Gam(G0*(h-1)+g,G0*(htil-1)+gtil) + cor(htil);
                    end
                end
                
            end
        end
    end
    Gam = (triu(Gam)+triu(Gam)') - eye(size(Gam,1)).*diag(Gam);
    GSE = reshape(diag(sqrt(Gam\V/Gam/N)),G0,H);
    
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