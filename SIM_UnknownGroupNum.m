%% Simulation Study #2 : unknown group number
% See Section 6.3 in the paper
% Alert: it's gonna take SUPER LONG to get the results
% because for each simulated panel, I run with different G^guess
% results are stored in the subfolder /output/SIM_UnknownGroup_...
% --------------------------- MODEL --------------------------------
% y_it = mu_i + phi_g*y_it-1 + beta_g*x_it + ep_it
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
% DGP
G0         = 3;               % true number of groups
parchoice  = 2;
par        = params(G0,parchoice);

% EST
FE         = 1;
inference  = 3;
H          = 6;               % h=0 is added in data preparation by default
nInit      = 100;
Gmax       = 8;

% SIM
nRep       = 500;
burn       = 100;
Ngrid      = [100, 200, 300];
Tgrid      = [100, 200, 300];

DGPsetup.par = par;
DGPsetup.H   = H;
DGPsetup.burn= burn;

%% True IRF
IR_true = zeros(G0,H+1);
for k = 1 : G0
    IR_true(k,:) = par(2,k)* (par(1,k) .^ [0:H]);
end

%% Simulation
for jj = 1:size(Ngrid,2)
        % initialization, creating temporary statistics holder
        Sim_Group   = cell(nRep,size(Tgrid,2));
        Sim_IRF     = cell(nRep,size(Tgrid,2));
        Sim_RMSE    = cell(nRep,size(Tgrid,2));
        Sim_OBJ     = cell(nRep,size(Tgrid,2));
        Sim_GEst    = nan(nRep,size(Tgrid,2));
        
        N   = Ngrid(jj);
        
        % assign membership
        if G0 == 2
            Ncut= N*[0.5 1];
        elseif G0==3
            Ncut= N*[0.3 0.6 1]; % for 3 groups
        elseif G0==4
            Ncut= N*[0.25 0.5 0.75 1];
        end
        
        id = 1:N;
        Gr0  = ones(N,1)*G0;
        for k=G0-1:-1:1
            Gr0 = Gr0 - ( id <=Ncut(k) )' *1;
        end
        DGPsetup.G   = Gr0;
        
        % create IR_TRUE for computing RMSE, N by H
        IR_TRUE = nan(N,H+1);
        for i =1:N
            IR_TRUE(i,:) = IR_true(Gr0(i),:);
        end
        
    for tt = 1:size(Tgrid,2)
        T   = Tgrid(tt);
        fprintf('Start working on grid [N=%d, T=%d] \n', N, T)
        %% Simulation starts here
        tic
        for iRep = 1:nRep            
            Sim = DGP(N,T,DGPsetup);
            
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
            
            
            Sim_Group{iRep,tt}   = Gr_EST;
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
    
    dataholder{jj,1} = Sim_Group;
    dataholder{jj,2} = Sim_IRF;
    dataholder{jj,3} = Sim_RMSE;
    dataholder{jj,4} = Sim_OBJ;
    dataholder{jj,5} = Sim_GEst;
    
    
end

save_name = strcat('output\SIM_UnknownGroup_K',num2str(G0),'_param',num2str(parchoice),'_ic.mat');
save(save_name);


%% RMSE
[reshape(mean(cell2mat(dataholder{1,3})),10,3)...
    reshape(mean(cell2mat(dataholder{2,3})),10,3) ...
    reshape(mean(cell2mat(dataholder{3,3})),10,3)]


%% Estmated Group Number - IC
[mean(dataholder{1,5}) mean(dataholder{2,5}) mean(dataholder{3,5})]


%% test other ic penalty
% for jj = 1:3
%     tmp = dataholder{jj,4};
%     for tt = 1:3
%         tmp1 = cell2mat(tmp(:,tt));
%         % baseline
%         pen = [1:8].*(H+1).*((Ngrid(jj)*(Tgrid(tt)-H))^(-0.2));
%         ictmp = tmp1+tmp1(:,end).*pen;
%         [~,Gtmp] = min(ictmp,[],2);
%         strcat('N=',num2str(Ngrid(jj)),', T=',num2str(Tgrid(tt)),', G=',num2str(mean(Gtmp)))
%     end
% end