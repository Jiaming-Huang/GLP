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

close all;
clear; clc;

rng(27);
addpath('./output');
addpath('./routines');
dataholder = cell(3,5);

%% Parameter Setting

% EST
FE         = 1;
H          = 6;               % h=0 is added in data preparation by default
nInit      = 100;
Gmax       = 8;

% SIM
nRep       = 500;
burn       = 100;
Ngrid      = [100, 200, 300];
Tgrid      = [100, 200, 300];

DGPsetup.H   = H;
DGPsetup.burn= burn;

%% Simulation
for jj = 1:size(Ngrid,2)
    % initialization, creating temporary statistics holder
    Sim_IRTrue  = cell(nRep,size(Tgrid,2));
    Sim_IRF     = cell(nRep,size(Tgrid,2));
    Sim_RMSE    = cell(nRep,size(Tgrid,2));
    Sim_OBJ     = cell(nRep,size(Tgrid,2));
    Sim_GEst    = nan(nRep,size(Tgrid,2));
    
    N   = Ngrid(jj);
    for tt = 1:size(Tgrid,2)
        T   = Tgrid(tt);
        fprintf('Start working on grid [N=%d, T=%d] \n', N, T)
        %% Simulation starts here
        tic
        for iRep = 1:nRep
            % simulation setup
            phi = (0.9-0.1)*rand(1,N);
            bet = (3-1)*rand(1,N);
            DGPsetup.par = [phi ;bet];
            % create IRF_TRUE for computing RMSE
            IR_TRUE = nan(N,H+1);
            for i =1:N
                IR_TRUE(i,:) = bet(i)*(phi(i).^[0:H]);
            end
            Sim_IRTrue{iRep,tt} = IR_TRUE;
            
            Sim = DGP_no_group(N,T,DGPsetup);
            
            %% G-LP Estimation
            % using st
            [b,~]=panel_LP(Sim.reg, FE);
            rmse0 = sqrt(mean( mean( (b - IR_TRUE).^2)));
            
            % get initial guess
            [bInit,weight,~] = ind_LP(Sim.reg);
            rmse1 = sqrt(mean( mean( (bInit - IR_TRUE).^2)));
            
            % glp
            [Gr_EST, GIRF, OBJ, IC] = GroupLPIV_Sim_Unknown_Group(Sim.reg, Gmax, nInit, bInit, weight, FE);
            
            rmse = nan(1,Gmax);
            for gr = 1:Gmax
                rmse(gr) = getRMSE(Gr_EST(:,gr), IR_TRUE, GIRF{1,gr});
            end            
            
            
            Sim_IRF{iRep, tt} = GIRF;
            Sim_RMSE{iRep, tt} = [rmse0 rmse rmse1];
            Sim_OBJ{iRep,tt} = OBJ;
            [~,gEst] = min(IC);
            Sim_GEst(iRep, tt) = gEst;

            fprintf('Simulation# %d: IC = %d \n', iRep,gEst)
        end
        t1 = toc;
        fprintf('Grid finished. Time used: %f seconds.\n', t1)
        
    end
    
    dataholder{jj,1} = Sim_IRTrue;
    dataholder{jj,2} = Sim_IRF;
    dataholder{jj,3} = Sim_RMSE;
    dataholder{jj,4} = Sim_OBJ;
    dataholder{jj,5} = Sim_GEst;
end

save('output\SIM_NoGroup.mat');

        
%% RMSE
[reshape(mean(cell2mat(dataholder{1,3})),10,3)...
    reshape(mean(cell2mat(dataholder{2,3})),10,3) ...
    reshape(mean(cell2mat(dataholder{3,3})),10,3)]


%% Estmated Group Number - IC
[mean(dataholder{1,5}) mean(dataholder{2,5}) mean(dataholder{3,5})]
