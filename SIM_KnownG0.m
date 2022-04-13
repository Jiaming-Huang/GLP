%% Simulation Study #1 : known group number
% See Section 6.2 in the paper
% results are stored in the subfolder /output/SIM_KnownGroup_...
% --------------------------- MODEL --------------------------------
% y_it = mu_i + phi_g*y_it-1 + beta_g*x_it + ep_it
% x_it = mu_i + pi*z_it + u_it
% mu_i ~ U(0,1)
% z_it, xi_it are iid N(0,1)
% u_it, ep_it are bivarate normal, with Sig=[1 0.3; 0.3 1]

%% Housekeeping
% close all;
% clear; clc;

rng(27);
% addpath('./output');
% addpath('./routines');

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
    IR_true(:,:,g,:) = par(2,g)* (par(1,g) .^ [0:H]);
end

%% Simulation
startAll = tic;
for jj = 1:NGridSize
    N   = Ngrid(jj);

    % initialization, creating temporary output holder
    % GLP - Default weighting matrix (asymV)
    GLP_GR       = cell(nRep,TGridSize); % GLP Gr_hat
    GLP_AC       = nan(nRep, TGridSize); % GLP classification accuracy
    GLP_IR       = cell(nRep,TGridSize); % GLP IR
    GLP_SE       = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSE      = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BR       = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CP       = cell(nRep,TGridSize); % GLP coverage probability

    % Benchmark: Infeasible GLP
    IGLP_IR      = cell(nRep,TGridSize); % IGLP IR
    IGLP_SE      = cell(nRep,TGridSize); % IGLP standard errors
    IGLP_MSE     = nan(nRep, TGridSize); % IGLP mean squared errors
    IGLP_BR      = nan(nRep,TGridSize); % IGLP confidence band ratio (relative to IND_LP)
    IGLP_CP      = cell(nRep,TGridSize); % IGLP coverage probability

    % Benchmark: Pool & Individual LPIV
    IND_MSE      = nan(nRep,TGridSize);  % IND_LP mean squared errors
    PAN_MSE      = nan(nRep,TGridSize);  % PAN mean squared errors

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
    Ng0          = sum(Gr0==[1:G0]);

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

            %% Benchmark: Panel LP-IV
            panOut = panel_LP(Sim.reg, FE);
            err2 = (panOut.IR - IR_TRUE).^2;
            PAN_MSE(iRep,tt) = mean(err2(:));

            %% Benchmark: Individual LP-IV
            indOut = ind_LP(Sim.reg);
            err2 = (indOut.IR - IR_TRUE).^2;
            IND_MSE(iRep,tt) = mean(err2(:));

            %% GLP Estimation - AsymV
            weight = indOut.asymV;
            [Gr, GIRF, GSE]   = GLP_SIM_KnownG0(Sim.reg, G0, IR_true, indOut.b(2,:,:,:), weight, FE, inference);
            [GLP_AC(iRep,tt), GLP_MSE(iRep,tt), GLP_BR(iRep,tt), ~, ~, Gr_re, GIRF_re, GSE_re] = eval_GroupLPIV([Gr0 Gr], IR_TRUE, GIRF, GSE, indOut.se(1:K,:,:,:));
            GLP_GR{iRep,tt}   = Gr_re;
            GLP_IR{iRep,tt}   = GIRF_re;
            GLP_SE{iRep,tt}   = GSE_re;
            Ubands            = GIRF_re + 1.96*GSE_re;
            Lbands            = GIRF_re - 1.96*GSE_re;
            GLP_CP{iRep,tt}   = (Ubands > IR_true) & (Lbands < IR_true);

%             for k=1:K
%                 figure;
%                 for g = 1:G0
%                     subplot(2,2,g);
%                     plot([0:H],squeeze(GIRF_re(k,:,g,:)),'k-');
%                     hold on;
%                     plot([0:H],squeeze(IR_true(k,:,g,:)),'b-');
%                     plot([0:H],squeeze(Ubands(k,:,g,:)),'r--');
%                     plot([0:H],squeeze(Lbands(k,:,g,:)),'r--');
%                     hold off
%                 end
%             end

            %% IGLP - Infeasible GLP (known Groups)
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

%             for k=1:K
%                 figure;
%                 for g = 1:G0
%                     subplot(2,2,g);
%                     plot([0:H],squeeze(IGIRF(k,:,g,:)),'k-');
%                     hold on;
%                     plot([0:H],squeeze(IR_true(k,:,g,:)),'b-');
%                     plot([0:H],squeeze(IUbands(k,:,g,:)),'r--');
%                     plot([0:H],squeeze(ILbands(k,:,g,:)),'r--');
%                     hold off
%                 end
%             end

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
    dataholder{jj,4}  = GLP_SE; % nRep x TGridSize x 3 cell
    % MSE
    dataholder{jj,5}  = [GLP_MSE, PAN_MSE, IND_MSE, IGLP_MSE]; % nRep x TGridSize x 4 matrix
    % Band ratios
    dataholder{jj,6}  = [GLP_BR, IGLP_BR]; % nRep x TGridSize x 2 matrix
    % Coverage probabilities
    dataholder{jj,7}  = GLP_CP; % nRep x TGridSize x G0 by H matrix
    dataholder{jj,8}  = IGLP_CP;

end
endAll = toc(startAll);
fprintf('Total execution time:: %f seconds.\n', endAll)

%% SAVE OUTPUT
save_name = strcat('output\SIM_KnownG',num2str(G0),'_param',num2str(parchoice),...
    '_FE.mat');
save(save_name);


%% TABLES
%% Accuracy
Accuracy = round([reshape(mean(dataholder{1,2}),TGridSize,1);
    reshape(mean(dataholder{2,2}),TGridSize,1);
    reshape(mean(dataholder{3,2}),TGridSize,1)],4);
disp(Accuracy)
%% RMSE for GLP (col 1), PAN (col 2), IND (col 3), IGLP (col 4)
RMSE = round(sqrt([reshape(mean(dataholder{1,5}),TGridSize,4);
    reshape(mean(dataholder{2,5}),TGridSize,4);
    reshape(mean(dataholder{3,5}),TGridSize,4)]),4);
disp(RMSE)
%% BR for - GLP IGLP
BR = round([reshape(mean(dataholder{1,6}),TGridSize,2);
    reshape(mean(dataholder{2,6}),TGridSize,2);
    reshape(mean(dataholder{3,6}),TGridSize,2)],4);
disp(BR)
%% Coverage GLP - AsymV
CP_GLP = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,7})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,7})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,7})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
disp(CP_GLP)
%% Coverage GLP - IGLP
CP_IGLP = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,8})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,8})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,8})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
disp(CP_IGLP)