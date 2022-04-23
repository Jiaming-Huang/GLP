%% Simulation Study #4 : known group number & large T
% See Section S1.1 in the supplemental material
% results are stored in the subfolder /output/SUPP/SIM_KnownGroup_LargeT
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
% EST
K          = 1;
FE         = 1;
inference  = 1;
H          = 6;               % h=0 is added in data preparation by default

% SIM
nRep       = 1000;
burn       = 100;
Ngrid      = [500, 1000, 1500];
T          = 300;

NGridSize  = size(Ngrid,2);
dataholder = cell(2,2,8);
DGPsetup.H   = H;
DGPsetup.burn= burn;

%% Simulation
startAll = tic;
for parchoice = 1:2
    for G0 = 2:3
        par          = params(G0,parchoice);
        DGPsetup.par = par;
        %% True IRF
        IR_true = zeros(K,1,G0,H+1);
        for g = 1 : G0
            IR_true(:,:,g,:) = par(2,g)* (par(1,g) .^ (0:H));
        end

        %% initialization, creating temporary statistics holder
        % GLP - Default weighting matrix (asymV)
        GLP_GR       = cell(nRep,NGridSize); % GLP Gr_hat
        GLP_AC       = nan(nRep, NGridSize); % GLP classification accuracy
        GLP_IR       = cell(nRep,NGridSize); % GLP IR
        GLP_SE       = cell(nRep,NGridSize); % GLP standard errors
        GLP_MSE      = nan(nRep, NGridSize); % GLP mean squared errors
        GLP_BR       = nan(nRep,NGridSize); % GLP confidence band ratio (relative to IND_LP)
        GLP_CP       = cell(nRep,NGridSize); % GLP coverage probability
        
        % Benchmark: Infeasible GLP
        IGLP_IR      = cell(nRep,NGridSize); % IGLP IR
        IGLP_SE      = cell(nRep,NGridSize); % IGLP standard errors
        IGLP_MSE     = nan(nRep, NGridSize); % IGLP mean squared errors
        IGLP_BR      = nan(nRep,NGridSize); % IGLP confidence band ratio (relative to IND_LP)
        IGLP_CP      = cell(nRep,NGridSize); % IGLP coverage probability

        % Benchmark: Pool & Individual LPIV
        IND_MSE      = nan(nRep,NGridSize);  % IND_LP mean squared errors
        PAN_MSE      = nan(nRep,NGridSize);  % PAN mean squared errors

        for jj = 1:NGridSize
            N   = Ngrid(jj);
            fprintf('Start working on G%d, Design %d, grid [N=%d, T=%d] \n', G0, parchoice, N, T)
            %% assign membership
            if G0 == 2
                Ncut= N*[0.5 1];
            elseif G0==3
                Ncut= N*[0.3 0.6 1]; % for 3 groups
            end
            
            id = 1:N;
            Gr0  = ones(N,1)*G0;
            for k=G0-1:-1:1
                Gr0 = Gr0 - ( id <=Ncut(k) )' *1;
            end
            DGPsetup.G   = Gr0;
            Ng0          = sum(Gr0==1:G0);

            %% create IRF_TRUE for computing RMSE
            IR_TRUE = nan(K,1,N,H+1);
            for i =1:N
                IR_TRUE(:,:,i,:) = IR_true(:,:,Gr0(i),:);
            end
            
            %% Simulation starts here
            startGrid = tic;
            parfor iRep = 1:nRep
                Sim = DGP(N,T,DGPsetup);
                
                %% Benchmark: Panel LP-IV
                panOut = panel_LP(Sim.reg, FE);
                err2 = (panOut.IR - IR_TRUE).^2;
                PAN_MSE(iRep,jj) = mean(err2(:));

                %% Benchmark: Individual LP-IV
                indOut = ind_LP(Sim.reg);
                err2 = (indOut.IR - IR_TRUE).^2;
                IND_MSE(iRep,jj) = mean(err2(:));

                %% GLP Estimation - AsymV
                weight = repmat(mean(indOut.v_hac,3),1,1,N,1);%indOut.v_hac;
                [Gr, GIRF, GSE]   = GLP_SIM_KnownG0(Sim.reg, G0, IR_true, indOut.b(2:end,:,:,:), weight, FE, inference);
                [GLP_AC(iRep,jj), GLP_MSE(iRep,jj), GLP_BR(iRep,jj), ~, ~, Gr_re, GIRF_re, GSE_re] = eval_GroupLPIV([Gr0 Gr], IR_TRUE, GIRF, GSE,indOut.se(1:K,:,:,:));
                GLP_GR{iRep,jj}   = Gr_re;
                GLP_IR{iRep,jj}   = GIRF_re;
                GLP_SE{iRep,jj}   = GSE_re;
                Ubands            = GIRF_re + 1.96*GSE_re;
                Lbands            = GIRF_re - 1.96*GSE_re;
                GLP_CP{iRep,jj}   = (Ubands > IR_true) & (Lbands < IR_true);

                %% IGLP - Infeasible GLP (known Groups)
                [IGIRF, IGSE, IUbands, ILbands] = GLP_SIM_Infeasible(Sim.reg, Gr0, FE);
                IGLP_IR{iRep,jj}     = IGIRF;
                IGLP_SE{iRep,jj}     = IGSE;
                
                IGIR_EST = nan(size(IR_TRUE));
                IGSE_EST = nan(size(IR_TRUE));
                for g = 1:G0
                    IGIR_EST(:,:,Gr0==g,:) = repmat(IGIRF(:,:,g,:),1,1,Ng0(g),1);
                    IGSE_EST(:,:,Gr0==g,:) = repmat(IGSE(:,:,g,:),1,1,Ng0(g),1);
                end
                err2              = (IGIR_EST - IR_TRUE).^2;
                IGLP_MSE(iRep,jj) = mean(err2(:));
                seRatio           = IGSE_EST./indOut.se(1:K,:,:,:);
                IGLP_BR(iRep,jj)  = mean(seRatio(:));
                IGLP_CP{iRep,jj}  = (IUbands > IR_true) & (ILbands < IR_true);

                fprintf('Iteration: %d \n', iRep)
            end
            endGrid = toc(startGrid);
            fprintf('Grid finished. Time used: %f seconds.\n', endGrid)
        end
        % Store outputs
        % Group membership
        dataholder{parchoice,G0-1,1} = GLP_GR;
        % Accuracy
        dataholder{parchoice,G0-1,2} = GLP_AC;
        % IRs
        dataholder{parchoice,G0-1,3} = GLP_IR;
        % Stanrd errors
        dataholder{parchoice,G0-1,4} = GLP_SE;
        % MSE
        dataholder{parchoice,G0-1,5} = [GLP_MSE, PAN_MSE, IND_MSE, IGLP_MSE];
        % Band ratios
        dataholder{parchoice,G0-1,6} = [GLP_BR, IGLP_BR];
        % Coverage probabilities
        dataholder{parchoice,G0-1,7} = GLP_CP;
        dataholder{parchoice,G0-1,8} = IGLP_CP;
        
    end
end
endAll = toc(startAll);
fprintf('Total execution time:: %f seconds.\n', endAll)
save_name = strcat('output\SUPP\SIM_SampSize_LargeT_KnownG0.mat');
save(save_name);


%% TABLES
%% Accuracy
Accuracy = [mean(dataholder{1,1,2})';mean(dataholder{1,2,2})';...
    mean(dataholder{2,1,2})';mean(dataholder{2,2,2})'];
disp(Accuracy);
%% Band Ratio: GLP (col 1) IGLP (col 2)
BR = [reshape(mean(dataholder{1,1,6}),NGridSize,2);...
    reshape(mean(dataholder{1,2,6}),NGridSize,2);...
    reshape(mean(dataholder{2,1,6}),NGridSize,2);...
    reshape(mean(dataholder{2,2,6}),NGridSize,2)];
disp(BR);
%% RMSE
RMSE = sqrt([reshape(mean(dataholder{1,1,5}),NGridSize,4);...
    reshape(mean(dataholder{1,2,5}),NGridSize,4);...
    reshape(mean(dataholder{2,1,5}),NGridSize,4);...
    reshape(mean(dataholder{2,2,5}),NGridSize,4)]);
disp(RMSE);
%% Coverage Probabilities: GLP
CP_GLP = [reshape(mean(mean(cell2mat(dataholder{1,1,7})),3),NGridSize,H+1);...
    reshape(mean(mean(cell2mat(dataholder{1,2,7})),3),NGridSize,H+1);...
    reshape(mean(mean(cell2mat(dataholder{2,1,7})),3),NGridSize,H+1);...
    reshape(mean(mean(cell2mat(dataholder{2,2,7})),3),NGridSize,H+1)];
disp(CP_GLP);
%% Coverage Probabilities: IGLP
CP_IGLP = [reshape(mean(mean(cell2mat(dataholder{1,1,8})),3),NGridSize,H+1);...
    reshape(mean(mean(cell2mat(dataholder{1,2,8})),3),NGridSize,H+1);...
    reshape(mean(mean(cell2mat(dataholder{2,1,8})),3),NGridSize,H+1);...
    reshape(mean(mean(cell2mat(dataholder{2,2,8})),3),NGridSize,H+1)];
disp(CP_IGLP);