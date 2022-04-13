%% Simulation Study #3 : no group structure
% See Section 6.4 in the paper
% Alert: it's gonna take SUPER LONG to get the results
% because for each simulated panel, I run with different G^guess
% results are stored in the subfolder /output/SIM_NoTrueGroup.mat
% --------------------------- MODEL --------------------------------
% y_it = mu_i + phi_i*y_it-1 + beta_i*x_it + ep_it
% x_it = mu_i + s_it + u_it
% z_it = s_it + xi_it
% mu_i ~ U(0,1)
% s_it, xi_it are iid N(0,1)
% u_it, ep_it are bivarate normal, with Sig=[1 0.3; 0.3 1]

%% Housekeeping
close all;
clear; clc;

rng(27);
% addpath('./output');
% addpath('./routines');

%% Parameter Setting
% EST
K          = 1;
FE         = 1;
H          = 6;               % h=0 is added in data preparation by default
nInit      = 50;
Gmax       = 8;

% SIM
nRep       = 500;
burn       = 100;
Ngrid      = [100, 200, 300];
Tgrid      = [100, 200, 300];

DGPsetup.K   = K;
DGPsetup.H   = H;
DGPsetup.burn= burn;

NGridSize  = size(Ngrid,2);
TGridSize  = size(Tgrid,2);
dataholder = cell(NGridSize,5);

%% Simulation
startAll = tic;
for jj = 1:NGridSize
    N   = Ngrid(jj);

    % initialization, creating temporary statistics holder
    SIM_IRTrue      = cell(nRep,size(Tgrid,2));
    GLP_GR          = cell(nRep,size(Tgrid,2)); % GLP Gr_hat
    GLP_IR          = cell(nRep,size(Tgrid,2)); % GLP IR
    GLP_OBJ         = cell(nRep,size(Tgrid,2)); % GLP objective functions
    GLP_GEst        = nan(nRep, size(Tgrid,2)); % estimated group number by IC
    SIM_MSE         = cell(nRep, size(Tgrid,2)); % GLP squared errors
    
    for tt = 1:TGridSize
        T   = Tgrid(tt);
        fprintf('Start working on grid [N=%d, T=%d] \n', N, T)
        %% Simulation starts here
        startGrid = tic;
        parfor iRep = 1:nRep
            %% Generate data
            Sim                 = DGP_NoGroup(N,T,DGPsetup);
            IR_TRUE             = Sim.IR_TRUE;
            SIM_IRTrue{iRep,tt} = IR_TRUE;
            
            %% Benchmark: Panel LP-IV
            panOut = panel_LP(Sim.reg, FE);
            err2 = (panOut.IR - IR_TRUE).^2;
            PAN_MSE = mean(err2(:));
            
            %% Benchmark: Individual LP-IV
            indOut  = ind_LP(Sim.reg);
            err2    = (indOut.IR - IR_TRUE).^2;
            IND_MSE = mean(err2(:));

            %% GLP Estimation
            weight = indOut.asymV;
            [Gr_EST, GIRF, OBJ, IC]   = GLP_SIM_UnknownG0(Sim.reg, Gmax, nInit, indOut.b, weight, FE);
            
            GLP_GR{iRep,tt}     = Gr_EST;
            GLP_IR{iRep,tt}     = GIRF;
            
            GLP_MSE     = nan(1,Gmax);
            GIR_EST = nan(size(IR_TRUE));
            for Ghat = 1:Gmax
                % map GIRF to K by H by N matrix
                Ng = sum(Gr_EST(:,Ghat)==[1:Ghat]);
                for g = 1:Ghat
                    girf = GIRF{1,Ghat};
                    GIR_EST(:,:,Gr_EST(:,Ghat)==g,:) = repmat(girf(:,:,g,:),1,1,Ng(g),1);
                end
                err2 = (GIR_EST - IR_TRUE).^2;
                GLP_MSE(1,Ghat) = mean(err2(:));
            end
            
            SIM_MSE{iRep, tt} = [PAN_MSE GLP_MSE IND_MSE];
            GLP_OBJ{iRep,tt} = OBJ;
            [~,gEst] = min(IC);
            GLP_GEst(iRep, tt) = gEst;
            
            fprintf('Simulation# %d: IC = %d \n', iRep, gEst)

        end
        endGrid = toc(startGrid);
        fprintf('Grid finished. Time used: %f seconds.\n', endGrid)
        
    end
    
    % Store outputs
    % True IRs
    dataholder{jj,1} = SIM_IRTrue;
    % Group membership
    dataholder{jj,2} = GLP_GR;
    % IRs
    dataholder{jj,3} = GLP_IR;
    % MSE
    dataholder{jj,4} = SIM_MSE;
    % Objective function
    dataholder{jj,5} = GLP_OBJ;
    % Estimated groups
    dataholder{jj,6} = GLP_GEst;
end

endAll = toc(startAll);
fprintf('Total execution time:: %f seconds.\n', endAll)

save('output\SIM_NoGroup.mat');

        
%% TABLES
%% RMSE for PAN (col 1), GLP (col 2:9), IND (col 10)
RMSE = round(sqrt([reshape(mean(cell2mat(dataholder{1,4})),10,TGridSize)';...
    reshape(mean(cell2mat(dataholder{2,4})),10,TGridSize)';...
    reshape(mean(cell2mat(dataholder{3,4})),10,TGridSize)']),4)';
disp(RMSE)
%% Estmated Group Number - IC
GEST = round([mean(dataholder{1,6}) mean(dataholder{2,6}) mean(dataholder{3,6})],1);
disp(GEST)