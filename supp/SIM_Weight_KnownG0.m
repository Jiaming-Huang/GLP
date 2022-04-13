%% Simulation Study #6 : known group number & Compare weighting matrix
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
inference  = 3;
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
dataholder = cell(NGridSize,13);

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
    % GLP - unit & horizon specific weighting matrix (asymV)
    GLP_GRih       = cell(nRep,TGridSize); % GLP Gr_hat
    GLP_ACih       = nan(nRep, TGridSize); % GLP classification accuracy
    GLP_IRih       = cell(nRep,TGridSize); % GLP IR
    GLP_SEih       = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSEih      = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BRih       = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CPih       = cell(nRep,TGridSize); % GLP coverage probability

    % GLP - horizon-specific weighting matrix (panel asymV)
    GLP_GRh       = cell(nRep,TGridSize); % GLP Gr_hat
    GLP_ACh       = nan(nRep, TGridSize); % GLP classification accuracy
    GLP_IRh       = cell(nRep,TGridSize); % GLP IR
    GLP_SEh       = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSEh      = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BRh       = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CPh       = cell(nRep,TGridSize); % GLP coverage probability

    % GLP - unit specific weighting matrix (mean asymV across h)
    GLP_GRi       = cell(nRep,TGridSize); % GLP Gr_hat
    GLP_ACi       = nan(nRep, TGridSize); % GLP classification accuracy
    GLP_IRi       = cell(nRep,TGridSize); % GLP IR
    GLP_SEi       = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSEi      = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BRi       = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CPi       = cell(nRep,TGridSize); % GLP coverage probability

    % GLP - unit & horizon specific weighting matrix for GLP
    %       IV for post GLP
    GLP_IRm       = cell(nRep,TGridSize); % GLP IR
    GLP_SEm       = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSEm      = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BRm       = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CPm       = cell(nRep,TGridSize); % GLP coverage probability

    % GLP - IV weighting
    GLP_GR_IV    = cell(nRep,TGridSize); % GLP Gr_hat
    GLP_AC_IV    = nan(nRep, TGridSize); % GLP classification accuracy
    GLP_IR_IV    = cell(nRep,TGridSize); % GLP IR
    GLP_SE_IV    = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSE_IV   = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BR_IV    = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CP_IV    = cell(nRep,TGridSize); % GLP coverage probability

    % GLP - 2SLS weighting
    GLP_GR_2SLS  = cell(nRep,TGridSize); % GLP Gr_hat
    GLP_AC_2SLS  = nan(nRep, TGridSize); % GLP classification accuracy
    GLP_IR_2SLS  = cell(nRep,TGridSize); % GLP IR
    GLP_SE_2SLS  = cell(nRep,TGridSize); % GLP standard errors
    GLP_MSE_2SLS = nan(nRep, TGridSize); % GLP mean squared errors
    GLP_BR_2SLS  = nan(nRep,TGridSize); % GLP confidence band ratio (relative to IND_LP)
    GLP_CP_2SLS  = cell(nRep,TGridSize); % GLP coverage probability

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

            %% GLP Estimation - unit&horizon specific weighting
            weight = indOut.asymV;
            [Grih, GIRFih, GIRFm, GSEih, GSEm] = GLP_SIM_KnownG0_Weight(Sim.reg, G0, IR_true, indOut.b(2,:,:,:), weight, FE);
            [GLP_ACih(iRep,tt), GLP_MSEih(iRep,tt), GLP_BRih(iRep,tt), ~, ~, Gr_reih, GIRF_reih, GSE_reih] = eval_GroupLPIV([Gr0 Grih], IR_TRUE, GIRFih, GSEih, indOut.se(1:K,:,:,:));
            GLP_GRih{iRep,tt}   = Gr_reih;
            GLP_IRih{iRep,tt}   = GIRF_reih;
            GLP_SEih{iRep,tt}   = GSE_reih;
            Ubandsih            = GIRF_reih + 1.96*GSE_reih;
            Lbandsih            = GIRF_reih - 1.96*GSE_reih;
            GLP_CPih{iRep,tt}   = (Ubandsih > IR_true) & (Lbandsih < IR_true);

            [~, GLP_MSEm(iRep,tt), GLP_BRm(iRep,tt), ~, ~, ~, GIRF_rem, GSE_rem] = eval_GroupLPIV([Gr0 Grih], IR_TRUE, GIRFm, GSEm, indOut.se(1:K,:,:,:));
            GLP_IRm{iRep,tt}   = GIRF_rem;
            GLP_SEm{iRep,tt}   = GSE_rem;
            Ubandsm            = GIRF_rem + 1.96*GSE_rem;
            Lbandsm            = GIRF_rem - 1.96*GSE_rem;
            GLP_CPm{iRep,tt}   = (Ubandsm > IR_true) & (Lbandsm < IR_true);

            %% GLP Estimation - horizon specific weighting
            weight = repmat(permute(panOut.asymV,[1,2,4,3]),1,1,N,1);%repmat(mean(indOut.asymV,3),1,1,N,1);
            [Grh, GIRFh, GSEh]   = GLP_SIM_KnownG0(Sim.reg, G0, IR_true, indOut.b(2,:,:,:), weight, FE, inference);
            [GLP_ACh(iRep,tt), GLP_MSEh(iRep,tt), GLP_BRh(iRep,tt), ~, ~, Gr_reh, GIRF_reh, GSE_reh] = eval_GroupLPIV([Gr0 Grh], IR_TRUE, GIRFh, GSEh, indOut.se(1:K,:,:,:));
            GLP_GRh{iRep,tt}   = Gr_reh;
            GLP_IRh{iRep,tt}   = GIRF_reh;
            GLP_SEh{iRep,tt}   = GSE_reh;
            Ubandsh            = GIRF_reh + 1.96*GSE_reh;
            Lbandsh            = GIRF_reh - 1.96*GSE_reh;
            GLP_CPh{iRep,tt}   = (Ubandsh > IR_true) & (Lbandsh < IR_true);

            %% GLP Estimation - unit specific weighting
            weight = repmat(mean(indOut.asymV,4),1,1,1,H+1);
            [Gri, GIRFi, GSEi]   = GLP_SIM_KnownG0(Sim.reg, G0, IR_true, indOut.b(2,:,:,:), weight, FE, inference);
            [GLP_ACi(iRep,tt), GLP_MSEi(iRep,tt), GLP_BRi(iRep,tt), ~, ~, Gr_rei, GIRF_rei, GSE_rei] = eval_GroupLPIV([Gr0 Gri], IR_TRUE, GIRFi, GSEi, indOut.se(1:K,:,:,:));
            GLP_GRi{iRep,tt}   = Gr_rei;
            GLP_IRi{iRep,tt}   = GIRF_rei;
            GLP_SEi{iRep,tt}   = GSE_rei;
            Ubandsi            = GIRF_rei + 1.96*GSE_rei;
            Lbandsi            = GIRF_rei - 1.96*GSE_rei;
            GLP_CPi{iRep,tt}   = (Ubandsi > IR_true) & (Lbandsi < IR_true);


            %% GLP Estimation - 2SLS
            weight = '2SLS';
            [Gr_2SLS, GIRF_2SLS, GSE_2SLS]   = GLP_SIM_KnownG0(Sim.reg, G0, IR_true, indOut.b(2,:,:,:), weight, FE, inference);
            [GLP_AC_2SLS(iRep,tt), GLP_MSE_2SLS(iRep,tt), GLP_BR_2SLS(iRep,tt), ~, ~, Gr_2SLS_re, GIRF_2SLS_re, GSE_2SLS_re] = eval_GroupLPIV([Gr0 Gr_2SLS], IR_TRUE, GIRF_2SLS, GSE_2SLS, indOut.se(1:K,:,:,:));
            GLP_GR_2SLS{iRep,tt}     = Gr_2SLS_re;
            GLP_IR_2SLS{iRep,tt}     = GIRF_2SLS_re;
            GLP_SE_2SLS{iRep,tt}     = GSE_2SLS_re;
            Ubands                   = GIRF_2SLS_re+1.96*GSE_2SLS_re;
            Lbands                   = GIRF_2SLS_re-1.96*GSE_2SLS_re;
            GLP_CP_2SLS{iRep,tt}     = (Ubands > IR_true) & (Lbands < IR_true);

            %% GLP Estimation - IV
            weight = 'IV';
            [Gr_IV, GIRF_IV, GSE_IV]   = GLP_SIM_KnownG0(Sim.reg, G0, IR_true, indOut.b(2,:,:,:), weight, FE, inference);
            [GLP_AC_IV(iRep,tt), GLP_MSE_IV(iRep,tt), GLP_BR_IV(iRep,tt), ~, ~, Gr_IV_re, GIRF_IV_re, GSE_IV_re] = eval_GroupLPIV([Gr0 Gr_IV], IR_TRUE, GIRF_IV, GSE_IV, indOut.se(1:K,:,:,:));
            GLP_GR_IV{iRep,tt}   = Gr_IV_re;
            GLP_IR_IV{iRep,tt}   = GIRF_IV_re;
            GLP_SE_IV{iRep,tt}   = GSE_IV_re;
            Ubands               = GIRF_IV_re+1.96*GSE_IV_re;
            Lbands               = GIRF_IV_re-1.96*GSE_IV_re;
            GLP_CP_IV{iRep,tt}   = (Ubands > IR_true) & (Lbands < IR_true);

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

            fprintf('Iteration: %d \n', iRep)
        end
        endGrid = toc(startGrid);
        fprintf('Grid finished. Time used: %f seconds.\n', endGrid)
    end

    %% STORE OUTPUT
    % Group membership
    dataholder{jj,1}  = [GLP_GRih, GLP_GRh, GLP_GRi, GLP_GR_2SLS, GLP_GR_IV]; % nRep x TGridSize x 3 cell
    % Accuracy
    dataholder{jj,2}  = [GLP_ACih, GLP_ACh, GLP_ACi, GLP_AC_2SLS, GLP_AC_IV]; % nRep x TGridSize x 3 matrix
    % IRs
    dataholder{jj,3}  = [GLP_IRih, GLP_IRh, GLP_IRi, GLP_IRm, GLP_IR_2SLS, GLP_IR_IV]; % nRep x TGridSize x 3 cell
    % Stanrd errors
    dataholder{jj,4}  = [GLP_SEih, GLP_SEh, GLP_SEi, GLP_SEm, GLP_SE_2SLS, GLP_SE_IV]; % nRep x TGridSize x 3 cell
    % MSE
    dataholder{jj,5}  = [GLP_MSEih, GLP_MSEh, GLP_MSEi, GLP_MSEm, GLP_MSE_2SLS, GLP_MSE_IV, PAN_MSE, IND_MSE, IGLP_MSE]; % nRep x TGridSize x 6 matrix
    % Band ratios
    dataholder{jj,6}  = [GLP_BRih, GLP_BRh, GLP_BRi, GLP_BRm, GLP_BR_2SLS, GLP_BR_IV, IGLP_BR]; % nRep x TGridSize x 4 matrix
    % Coverage probabilities
    dataholder{jj,7}  = GLP_CPih; % nRep x TGridSize x G0 by H matrix
    dataholder{jj,8}  = GLP_CPh; % nRep x TGridSize x G0 by H matrix
    dataholder{jj,9}  = GLP_CPi; % nRep x TGridSize x G0 by H matrix
    dataholder{jj,10}  = GLP_CPm;
    dataholder{jj,11}  = GLP_CP_2SLS;
    dataholder{jj,12} = GLP_CP_IV;
    dataholder{jj,13} = IGLP_CP;

end
endAll = toc(startAll);
fprintf('Total execution time:: %f seconds.\n', endAll)

%% SAVE OUTPUT
save_name = strcat('output\SUPP\SIM_Weight_KnownG',num2str(G0),'_param',num2str(parchoice),...
    '_FE.mat');
save(save_name);


%% TABLES
%% Accuracy for AsymV (col 1), IV (col 2), 2SLS (col 3)
Accuracy = round([reshape(mean(dataholder{1,2}),TGridSize,5);
    reshape(mean(dataholder{2,2}),TGridSize,5);
    reshape(mean(dataholder{3,2}),TGridSize,5)],4);
disp(Accuracy)
%% RMSE for AsymV (col 1), IV (col 2) and 2SLS (col 3), PAN (col 4), IND
% (col 5), IGLP (col 6)
RMSE = round(sqrt([reshape(mean(dataholder{1,5}),TGridSize,9);
    reshape(mean(dataholder{2,5}),TGridSize,9);
    reshape(mean(dataholder{3,5}),TGridSize,9)]),5);
disp(RMSE)
%% BR for - AsymV, IV, 2SLS IGLP
BR = round([reshape(mean(dataholder{1,6}),TGridSize,7);
    reshape(mean(dataholder{2,6}),TGridSize,7);
    reshape(mean(dataholder{3,6}),TGridSize,7)],4);
disp(BR)
%% Coverage Rates
% %% Coverage GLP - AsymV
% CP_GLP = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,7})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{2,7})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{3,7})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
% disp(CP_GLP)
% %% Coverage GLP - IV
% CP_IV = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,8})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{2,8})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{3,8})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
% disp(CP_IV)
% %% Coverage GLP - 2SLS
% CP_2SLS = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,9})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{2,9})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{3,9})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
% disp(CP_2SLS)
% %% Coverage GLP - IGLP
% CP_IGLP = array2table(round([reshape(mean(mean(cell2mat(dataholder{1,10})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{2,10})),3),TGridSize,H+1);
%     reshape(mean(mean(cell2mat(dataholder{3,10})),3),TGridSize,H+1)],4),'VariableNames',{'h=0','h=1','h=2','h=3','h=4','h=5','h=6'});
% disp(CP_IGLP)

CP_GLPih = round([reshape(mean(mean(cell2mat(dataholder{1,7})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,7})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,7})),3),TGridSize,H+1)],4);
CP_GLPh = round([reshape(mean(mean(cell2mat(dataholder{1,8})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,8})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,8})),3),TGridSize,H+1)],4);
CP_GLPi = round([reshape(mean(mean(cell2mat(dataholder{1,9})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,9})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,9})),3),TGridSize,H+1)],4);
CP_GLPm = round([reshape(mean(mean(cell2mat(dataholder{1,10})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,10})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,10})),3),TGridSize,H+1)],4);
CP_2SLS = round([reshape(mean(mean(cell2mat(dataholder{1,11})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,11})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,11})),3),TGridSize,H+1)],4);
CP_IV = round([reshape(mean(mean(cell2mat(dataholder{1,12})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,12})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,12})),3),TGridSize,H+1)],4);
CP_IGLP = round([reshape(mean(mean(cell2mat(dataholder{1,13})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{2,13})),3),TGridSize,H+1);
    reshape(mean(mean(cell2mat(dataholder{3,13})),3),TGridSize,H+1)],4);
CP_Stacked = [];
for i = 1:size(CP_GLPih,1)
    CP_Stacked = [CP_Stacked;CP_GLPih(i,:);CP_GLPh(i,:);CP_GLPi(i,:);CP_GLPm(i,:);CP_2SLS(i,:);CP_IV(i,:);CP_IGLP(i,:)];
end
CP_Stacked