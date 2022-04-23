%% Simulation Study #7 : compare GMM obj
% See Section S1.3 in the supplemental material
% results are stored in the subfolder
% /output/SUPP/SIM_CompareGMMOBJ

% --------------------------- MODEL --------------------------------
% y_it = mu_i + beta*x_it + ep_it
% x_it = mu_i + s_it + u_it
% z_it = s_it + xi_it
% mu_i ~ U(0,1)
% s_it, xi_it are iid N(0,1)
% u_it, ep_it are bivarate normal, with Sig=[1 0.3; 0.3 1]

%% Housekeeping
% close all;
% clear; clc;

rng(27);
% addpath('../output');
% addpath('../routines');

%% Parameter Setting
% DGP
% G0         = 3;               % true number of groups
% parchoice  = 1;
par        = params(G0,parchoice);

% EST
K          = 1;
FE         = 1;
inference  = 1;               % Large T inference, re-estimate with IV weights
H          = 6;               % h=0 is added in data preparation by default

% SIM
nRep       = 1000;
burn       = 100;
Ngrid      = [100, 200, 300];
Tgrid      = [100, 200, 300];

DGPsetup.par = par;
DGPsetup.H   = H;
DGPsetup.burn= burn;

NGridSize  = size(Ngrid,2);
TGridSize  = size(Tgrid,2);
dataholder = cell(NGridSize,8);

%% True IRF
IR_true = zeros(K,1,G0,H+1);
for g = 1 : G0
    IR_true(:,:,g,:) = par(2,g)* (par(1,g) .^ (0:H));
end

%% Simulation
startAll = tic;
for jj = 1:NGridSize
    N   = Ngrid(jj);

    % initialization, creating temporary output holder
    % Benchmark: Infeasible GLP as fully Pooled panel GMM
    IGLP_IR      = cell(nRep,TGridSize); % IGLP IR
    IGLP_SE      = cell(nRep,TGridSize); % IGLP standard errors
    IGLP_MSE     = nan(nRep, TGridSize); % IGLP mean squared errors
    IGLP_BR      = nan(nRep,TGridSize); % IGLP confidence band ratio (relative to IND_LP)
    IGLP_CP      = cell(nRep,TGridSize); % IGLP coverage probability

    % Benchmark: Infeasible GLP with our GMM criterion
    IGLP_IR1      = cell(nRep,TGridSize); % IGLP IR
    IGLP_SE1      = cell(nRep,TGridSize); % IGLP standard errors
    IGLP_MSE1     = nan(nRep, TGridSize); % IGLP mean squared errors
    IGLP_BR1      = nan(nRep,TGridSize); % IGLP confidence band ratio (relative to IND_LP)
    IGLP_CP1      = cell(nRep,TGridSize); % IGLP coverage probability
    
    % assign true membership
    if G0 == 2
        Ncut = N*[0.5 1];
    elseif G0==3
        Ncut = N*[0.3 0.6 1]; % for 3 groups
    end

    id = 1:N;
    Gr0  = ones(N,1)*G0;
    for k=G0-1:-1:1
        Gr0 = Gr0 - ( id <=Ncut(k) )' *1;
    end
    DGPsetup.G   = Gr0;
    Ng0          = sum(Gr0==1:G0);

    % create IRF_TRUE for computing RMSE
    IR_TRUE = nan(K,1,N,H+1);
    for i =1:N
        IR_TRUE(:,:,i,:) = IR_true(:,:,Gr0(i),:);
    end

    for tt = 1:TGridSize
        T   = Tgrid(tt);
        fprintf('Start working on grid [N=%d, T=%d] \n', N, T)
        %% Simulation starts here
        startGrid = tic;
        parfor iRep = 1:nRep
            Sim = DGP(N,T,DGPsetup);

            %% Benchmark: Individual LP-IV
            indOut = ind_LP(Sim.reg);
            
            %% IGLP - Infeasible GLP (Conventional GMM)
            [IGIRF, IGSE, IUbands, ILbands] = GLP_SIM_Infeasible(Sim.reg, Gr0, FE);
            IGLP_IR{iRep,tt}     = IGIRF;
            IGLP_SE{iRep,tt}     = IGSE;
            
            IGIR_EST = nan(size(IR_TRUE));
            IGSE_EST = nan(size(IR_TRUE));
            for g = 1:G0
                IGIR_EST(:,:,Gr0==g,:) = repmat(IGIRF(:,:,g,:),1,1,Ng0(g),1);
                IGSE_EST(:,:,Gr0==g,:) = repmat(IGSE(:,:,g,:),1,1,Ng0(g),1);
            end
            err2              = (IGIR_EST - IR_TRUE).^2;
            IGLP_MSE(iRep,tt) = mean(err2(:));
            seRatio           = IGSE_EST./indOut.se(1:K,:,:,:);
            IGLP_BR(iRep,tt)  = mean(seRatio(:));
            IGLP_CP{iRep,tt}  = (IUbands > IR_true) & (ILbands < IR_true);

            %% IGLP - Infeasible GLP (our GMM)
            [IGIRF1, IGSE1, IUbands1, ILbands1] = GLP_SIM_Infeasible1(Sim.reg, Gr0, FE);
            IGLP_IR1{iRep,tt}     = IGIRF1;
            IGLP_SE1{iRep,tt}     = IGSE1;
            
            IGIR_EST1 = nan(size(IR_TRUE));
            IGSE_EST1 = nan(size(IR_TRUE));
            for g = 1:G0
                IGIR_EST1(:,:,Gr0==g,:) = repmat(IGIRF1(:,:,g,:),1,1,Ng0(g),1);
                IGSE_EST1(:,:,Gr0==g,:) = repmat(IGSE1(:,:,g,:),1,1,Ng0(g),1);
            end
            err2              = (IGIR_EST1 - IR_TRUE).^2;
            IGLP_MSE1(iRep,tt) = mean(err2(:));
            seRatio           = IGSE_EST1./indOut.se(1:K,:,:,:);
            IGLP_BR1(iRep,tt)  = mean(seRatio(:));
            IGLP_CP1{iRep,tt}  = (IUbands1 > IR_true) & (ILbands1 < IR_true);
           
            fprintf('Iteration: %d \n', iRep)
        end
        endGrid = toc(startGrid);
        fprintf('Grid finished. Time used: %f seconds.\n', endGrid)
    end

    % Store outputs
    % Group membership
    dataholder{jj,1}  = [IGLP_IR IGLP_IR1]; % nRep x TGridSize x 1 cell
    % Accuracy
    dataholder{jj,2}  = [IGLP_SE IGLP_SE1]; % nRep x TGridSize x 1 matrix
    % IRs
    dataholder{jj,3}  = [IGLP_MSE IGLP_MSE1]; % nRep x TGridSize x 3 cell
    % Stanrd errors
    dataholder{jj,4}  = [IGLP_BR IGLP_BR1]; % nRep x TGridSize x 3 cell
    % MSE
    dataholder{jj,5}  = IGLP_CP; % nRep x TGridSize x G0 by H matrix
    dataholder{jj,6}  = IGLP_CP1;

end
endAll = toc(startAll);
fprintf('Total execution time:: %f seconds.\n', endAll)

%% SAVE OUTPUT
save_name = strcat('output\SIM_CompareGMM_G',num2str(G0),'_param',num2str(parchoice),...
    '_FE.mat');
save(save_name);

%% TABLES
%% RMSE for IGLP conventional (col 1), IGLP (col 2)
RMSE = round(sqrt([reshape(mean(dataholder{1,3}),TGridSize,2);
    reshape(mean(dataholder{2,3}),TGridSize,2);
    reshape(mean(dataholder{3,3}),TGridSize,2)]),4);
disp(RMSE)
%% BR for - IGLP conventional (col 1), IGLP (col 2)
BR = round([reshape(mean(dataholder{1,4}),TGridSize,2);
    reshape(mean(dataholder{2,4}),TGridSize,2);
    reshape(mean(dataholder{3,4}),TGridSize,2)],4);
disp(BR)
%% Coverage IGLP conventional 
CP_IGLP = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,5})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,5})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,5})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
disp(CP_IGLP)
%% Coverage GLP - IGLP
CP_IGLP1 = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,6})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,6})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,6})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
disp(CP_IGLP1)