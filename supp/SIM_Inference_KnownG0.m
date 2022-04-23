%% Simulation Study #1 : known group number
% See Section 6.2 in the paper
% results are stored in the subfolder /output/SIM_KnownGroup_...
% --------------------------- MODEL --------------------------------
% y_it = mu_i + phi_g*y_it-1 + beta_g*x_it + ep_it
% x_it = mu_i + s_it + u_it
% z_it = s_it + xi_it
% mu_i ~ U(0,1)
% s_it, xi_it are iid N(0,1)
% u_it, ep_it are bivarate normal, with Sig=[1 0.3; 0.3 1]

%% Housekeeping
% close all;
% clear; clc;

rng(27);
addpath('../output');
addpath('../routines');

%% Parameter Setting
% DGP
% G0         = 3;               % true number of groups
% parchoice  = 1;
par        = params(G0,parchoice);

% EST
K          = 1;
FE         = 1;
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
    % GLP - Large T
    GLP_GR       = cell(nRep,TGridSize); % GLP Gr_hat
    GLP_AC       = nan(nRep, TGridSize); % GLP classification accuracy
    GLP_IR       = cell(nRep,TGridSize); % GLP IR
    GLP_SE       = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSE      = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BR       = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CP       = cell(nRep,TGridSize); % GLP coverage probability

    % GLP - Fixed T
    GLP_SE_FT       = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSE_FT      = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BR_FT       = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CP_FT       = cell(nRep,TGridSize); % GLP coverage probability

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
        for iRep = 1:nRep
            Sim = DGP(N,T,DGPsetup);

            %% Benchmark: Individual LP-IV
            indOut = ind_LP(Sim.reg);

            %% GLP Estimation - AsymV
            weight = repmat(mean(indOut.v_hac,3),1,1,N,1);%indOut.v_hac;
            [Gr, GIRF, GSE, GSE_FT]   = GLP_SIM_KnownG0_Inference(Sim.reg, G0, IR_true, indOut.b(2:end,:,:,:), weight, FE);
            [GLP_AC(iRep,tt), GLP_MSE(iRep,tt), GLP_BR(iRep,tt), Permutation, Ng, Gr_re, GIRF_re, GSE_re] = eval_GroupLPIV([Gr0 Gr], IR_TRUE, GIRF, GSE, indOut.se(1:K,:,:,:));
            GLP_GR{iRep,tt}   = Gr_re;
            GLP_IR{iRep,tt}   = GIRF_re;
            GLP_SE{iRep,tt}   = GSE_re;
            % get it for fixed T
            GSE_re_FT            = GSE_FT(:,:,Permutation,:);
            GLP_SE_FT{iRep,tt}   = GSE_re_FT;
            GSE_EST = nan(size(IR_TRUE));
            for g = 1:G0
                GSE_EST(:,:,Gr_re==g,:) = repmat(GSE_re_FT(:,:,g,:),1,1,Ng(g),1);
            end
            seRatio = GSE_EST./indOut.se(1:K,:,:,:);
            GLP_BR_FT(iRep,tt) = mean(seRatio(:));
            
            Ubands              = GIRF_re + 1.96*GSE_re;
            Lbands              = GIRF_re - 1.96*GSE_re;
            Ubands_FT           = GIRF_re + 1.96*GSE_re_FT;
            Lbands_FT           = GIRF_re - 1.96*GSE_re_FT;
            GLP_CP{iRep,tt}     = (Ubands > IR_true) & (Lbands < IR_true);
            GLP_CP_FT{iRep,tt}  = (Ubands_FT > IR_true) & (Lbands_FT < IR_true);

            fprintf('Iteration: %d \n', iRep)
        end
        endGrid = toc(startGrid);
        fprintf('Grid finished. Time used: %f seconds.\n', endGrid)
    end

    % Store outputs
    % Group membership
    dataholder{jj,1}  = GLP_GR; % nRep x TGridSize x 1 cell
    % Accuracy
    dataholder{jj,2}  = GLP_AC; % nRep x TGridSize x 1 matrix
    % IRs
    dataholder{jj,3}  = GLP_IR; % nRep x TGridSize x 3 cell
    % Stanrd errors
    dataholder{jj,4}  = [GLP_SE, GLP_SE_FT]; % nRep x TGridSize x 3 cell
    % MSE
    dataholder{jj,5}  = GLP_MSE; % nRep x TGridSize x 4 matrix
    % Band ratios
    dataholder{jj,6}  = [GLP_BR, GLP_BR_FT]; % nRep x TGridSize x 2 matrix
    % Coverage probabilities
    dataholder{jj,7}  = GLP_CP; % nRep x TGridSize x G0 by H matrix
    dataholder{jj,8}  = GLP_CP_FT;

end
endAll = toc(startAll);
fprintf('Total execution time:: %f seconds.\n', endAll)

%% SAVE OUTPUT
save_name = strcat('output\SUPP\SIM_Inference_KnownG',num2str(G0),'_param',num2str(parchoice),...
    '_FE.mat');
save(save_name);


%% TABLES
%% BR for - GLP GLP_FT GLP_PT
BR = round([reshape(mean(dataholder{1,6}),TGridSize,2);
    reshape(mean(dataholder{2,6}),TGridSize,2);
    reshape(mean(dataholder{3,6}),TGridSize,2)],4);
disp(BR)
%% Coverage GLP - base
CP_GLP = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,7})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,7})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,7})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
disp(CP_GLP)
%% Coverage GLP - Fixed T
CP_GLP_FT = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,8})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,8})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,8})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
disp(CP_GLP_FT)