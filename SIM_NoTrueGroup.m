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
TSLS       = 1; 
FE         = 1;
H          = 6;               % h=0 is added in data preparation by default
ninit      = 100;

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
        Sim_Group   = cell(nRep,size(Tgrid,2));
        Sim_Qpath   = cell(nRep,size(Tgrid,2));
        Sim_IRF     = cell(nRep,size(Tgrid,2));
        Sim_GSE     = cell(nRep,size(Tgrid,2));
        Sim_bias    = cell(nRep,size(Tgrid,2));
        Sim_RMSE    = cell(nRep,size(Tgrid,2));
        IND_rmse        = nan(nRep,size(Tgrid,2));
        PAN_rmse        = nan(nRep,size(Tgrid,2));
        IND_SE          = cell(nRep,size(Tgrid,2));
        N   = Ngrid(jj);
        
        G   = [1:N]';
        % simulation setup
        phi = (0.9-0.1)*rand(1,N);
        bet = (3-1)*rand(1,N);
        DGPsetup.par = [phi ;bet];
        % create IRF_TRUE for computing RMSE
        IRF_TRUE = nan(N,H+1);
        for i =1:N
            IRF_TRUE(i,:) = bet(i)*(phi(i).^[0:H]);
        end
    for tt = 1:size(Tgrid,2)
        T   = Tgrid(tt);
        fprintf('Start working on grid [N=%d, T=%d] \n', N, T)
        %% Simulation starts here
        tic
        for iRep = 1:nRep
            Sim = DGP_no_group(N,T,DGPsetup);
            
            %% G-LP Estimation
            get initial guess
            [b,se,~]=ind_LP(Sim.reg);
            IND_rmse(iRep,tt) = sqrt(mean( mean( (b - IRF_TRUE).^2)));
            IND_SE{iRep,tt} = mean(se);
            binit = b;
            
            [G1, GIRF1, GSE1, Qpath1, ~, ~]   = GroupLPIV(Sim.reg, 1, ninit, TSLS, FE,binit);
            [G2, GIRF2, GSE2, Qpath2, ~, ~]   = GroupLPIV(Sim.reg, 2, ninit, TSLS, FE,binit);
            [G3, GIRF3, GSE3, Qpath3, ~, ~]   = GroupLPIV(Sim.reg, 3, ninit, TSLS, FE,binit);
            [G4, GIRF4, GSE4, Qpath4, ~, ~]   = GroupLPIV(Sim.reg, 4, ninit, TSLS, FE,binit);
            [G5, GIRF5, GSE5, Qpath5, ~, ~]   = GroupLPIV(Sim.reg, 5, ninit, TSLS, FE,binit);
            [G6, GIRF6, GSE6, Qpath6, ~, ~]   = GroupLPIV(Sim.reg, 6, ninit, TSLS, FE,binit);
            [G7, GIRF7, GSE7, Qpath7, ~, ~]   = GroupLPIV(Sim.reg, 7, ninit, TSLS, FE,binit);
            [G8, GIRF8, GSE8, Qpath8, ~, ~]   = GroupLPIV(Sim.reg, 8, ninit, TSLS, FE,binit);

            [~, rmse1, ~, p1, ~]    = eval_GroupLPIV_noTrue([G G1], IRF_TRUE, GIRF1);
            [~, rmse2, ~, p2, ~]    = eval_GroupLPIV_noTrue([G G2], IRF_TRUE, GIRF2);
            [~, rmse3, ~, p3, ~]    = eval_GroupLPIV_noTrue([G G3], IRF_TRUE, GIRF3);
            [~, rmse4, ~, p4, ~]    = eval_GroupLPIV_noTrue([G G4], IRF_TRUE, GIRF4);
            [~, rmse5, ~, p5, ~]    = eval_GroupLPIV_noTrue([G G5], IRF_TRUE, GIRF5);
            [~, rmse6, ~, p6, ~]    = eval_GroupLPIV_noTrue([G G6], IRF_TRUE, GIRF6);
            [~, rmse7, ~, p7, ~]    = eval_GroupLPIV_noTrue([G G7], IRF_TRUE, GIRF7);
            [~, rmse8, ~, p8, ~]    = eval_GroupLPIV_noTrue([G G8], IRF_TRUE, GIRF8);
            
            
            Sim_Group{iRep,tt}   = [G1 G2 G3 G4 G5 G6 G7 G8];
            Sim_Qpath{iRep,tt}   = [Qpath1*ones(ninit,1) Qpath2 Qpath3 Qpath4 Qpath5 Qpath6 Qpath7 Qpath8];
            Sim_IRF{iRep, tt} = [GIRF1(p1,:);GIRF2(p2,:);GIRF3(p3,:);GIRF4(p4,:);GIRF5(p5,:);GIRF6(p6,:);GIRF7(p7,:);GIRF8(p8,:)];
            Sim_GSE{iRep, tt} = [GSE1(p1,:);GSE2(p2,:);GSE3(p3,:);GSE4(p4,:);GSE5(p5,:);GSE6(p6,:);GSE7(p7,:);GSE8(p8,:)];
            Sim_RMSE{iRep, tt} = [rmse1 rmse2 rmse3 rmse4 rmse5 rmse6 rmse7 rmse8];
            
            %% Benchmark: Panel LP
            [b,~,~]=panel_LP1(Sim.reg, FE);
            PAN_rmse(iRep,tt) = sqrt(mean( mean( (b - IRF_TRUE).^2)));
        end
        t1 = toc;
        fprintf('Grid finished. Time used: %f seconds.\n', t1)
        
    end
    
    dataholder{jj,1} = Sim_Group;
    dataholder{jj,2} = Sim_Qpath;
    dataholder{jj,3} = Sim_IRF;
    dataholder{jj,4} = Sim_GSE;
    dataholder{jj,5} = Sim_RMSE;
    dataholder{jj,6} = IND_rmse;
    dataholder{jj,7} = PAN_rmse;
    dataholder{jj,8} = IND_SE;
    
end

save('output\SIM_NoTrueGroup.mat');

%% for tables
% GLP RMSE
[reshape(mean(cell2mat(dataholder{1,5})),8,3) reshape(mean(cell2mat(dataholder{2,5})),8,3) reshape(mean(cell2mat(dataholder{3,5})),8,3)]
    
% IND RMSE
[mean(dataholder{1,6}) mean(dataholder{2,6}) mean(dataholder{3,6})]

% PAN RMSE
[mean(dataholder{1,7}) mean(dataholder{2,7}) mean(dataholder{3,7})]


% BIC
% change jj to get results for different sample sizes
jj = 3;
N = Ngrid(jj);

tmp = dataholder{jj,2};
for t = 1:3
    for iRep = 1:500
        tmp{iRep,t} = min(tmp{iRep,t});
    end
end

BIC = nan(500,3);
for t = 1:3
    T = Tgrid(t)-H;
    for iRep = 1:500
        tt = tmp{iRep,t};
        bic = tt + tt(8).*[1:8] * (N+T)/(N*T) * log(N*T);
        [~,BIC(iRep,t)] = min(bic);
    end
end
mean(BIC)
