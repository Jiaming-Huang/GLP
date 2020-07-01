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
K0         = 3;               % true number of groups
parchoice  = 1;
par        = params(K0,parchoice);
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

DGPsetup.par = par;
DGPsetup.H   = H;
DGPsetup.burn= burn;

%% True IRF
IRF_true = zeros(K0,H+1);
for k = 1 : K0
    IRF_true(k,:) = par(2,k)* (par(1,k) .^ [0:H]);
end


%% Simulation
for jj = 1:size(Ngrid,2)
        % initialization, creating temporary statistics holder
        Sim_Group   = cell(nRep,size(Tgrid,2));
        Sim_Qpath   = cell(nRep,size(Tgrid,2));
        Sim_IRF     = cell(nRep,size(Tgrid,2));
        Sim_GSE     = cell(nRep,size(Tgrid,2));
        Sim_bias    = cell(nRep,size(Tgrid,2));
        Sim_RMSE    = cell(nRep,size(Tgrid,2));
        IND_rmse    = nan(nRep,size(Tgrid,2));
        N   = Ngrid(jj);
        if K0 == 2
            Ncut= N*[0.5 1];
        elseif K0==3
            Ncut= N*[0.3 0.6 1]; % for 3 groups
        elseif K0==4
            Ncut= N*[0.25 0.5 0.75 1];
        end
        
        % assign membership
        id = 1:N;
        G  = ones(N,1)*K0;
        for k=K0-1:-1:1
            G = G - ( id <=Ncut(k) )' *1;
        end
        DGPsetup.G   = G;
        
        % create IRF_TRUE for computing RMSE
        IRF_TRUE = nan(N,H+1);
        for i =1:N
            IRF_TRUE(i,:) = IRF_true(G(i),:);
        end
        
    for tt = 1:size(Tgrid,2)
        T   = Tgrid(tt);
        fprintf('Start working on grid [N=%d, T=%d] \n', N, T)
        %% Simulation starts here
        tic
        for iRep = 1:nRep            
            Sim = DGP(N,T,DGPsetup);
            
            %% G-LP Estimation
            % get initial guess
            [binit,~,~]=ind_LP(Sim.reg);
            IND_rmse(iRep,tt) = sqrt(mean( mean( (binit - IRF_TRUE).^2)));
            
            [G1, GIRF1, GSE1, Qpath1, ~, ~]   = GroupLPIV(Sim.reg, 1, ninit, TSLS, FE,binit);
            [G2, GIRF2, GSE2, Qpath2, ~, ~]   = GroupLPIV(Sim.reg, 2, ninit, TSLS, FE,binit);
            [G3, GIRF3, GSE3, Qpath3, ~, ~]   = GroupLPIV(Sim.reg, 3, ninit, TSLS, FE,binit);
            [G4, GIRF4, GSE4, Qpath4, ~, ~]   = GroupLPIV(Sim.reg, 4, ninit, TSLS, FE,binit);
            [G5, GIRF5, GSE5, Qpath5, ~, ~]   = GroupLPIV(Sim.reg, 5, ninit, TSLS, FE,binit);
            [G6, GIRF6, GSE6, Qpath6, ~, ~]   = GroupLPIV(Sim.reg, 6, ninit, TSLS, FE,binit);
            [G7, GIRF7, GSE7, Qpath7, ~, ~]   = GroupLPIV(Sim.reg, 7, ninit, TSLS, FE,binit);
            [G8, GIRF8, GSE8, Qpath8, ~, ~]   = GroupLPIV(Sim.reg, 8, ninit, TSLS, FE,binit);
            
            [~, rmse1, ~, p1, ~]    = eval_GroupLPIV([G G1], IRF_true, GIRF1);
            [~, rmse2, ~, p2, ~]    = eval_GroupLPIV([G G2], IRF_true, GIRF2);
            [~, rmse3, ~, p3, ~]    = eval_GroupLPIV([G G3], IRF_true, GIRF3);
            [~, rmse4, ~, p4, ~]    = eval_GroupLPIV([G G4], IRF_true, GIRF4);
            [~, rmse5, ~, p5, ~]    = eval_GroupLPIV([G G5], IRF_true, GIRF5);
            [~, rmse6, ~, p6, ~]    = eval_GroupLPIV([G G6], IRF_true, GIRF6);
            [~, rmse7, ~, p7, ~]    = eval_GroupLPIV([G G7], IRF_true, GIRF7);
            [~, rmse8, ~, p8, ~]    = eval_GroupLPIV([G G8], IRF_true, GIRF8);
            
            Sim_Group{iRep,tt}   = [G1 G2 G3 G4 G5 G6 G7 G8];
            Sim_Qpath{iRep,tt}   = [Qpath1*ones(ninit,1) Qpath2 Qpath3 Qpath4 Qpath5 Qpath6 Qpath7 Qpath8];
            Sim_IRF{iRep, tt} = [GIRF1(p1,:);GIRF2(p2,:);GIRF3(p3,:);GIRF4(p4,:);GIRF5(p5,:);GIRF6(p6,:);GIRF7(p7,:);GIRF8(p8,:)];
            Sim_GSE{iRep, tt} = [GSE1(p1,:);GSE2(p2,:);GSE3(p3,:);GSE4(p4,:);GSE5(p5,:);GSE6(p6,:);GSE7(p7,:);GSE8(p8,:)];
            Sim_RMSE{iRep, tt} = [rmse1 rmse2 rmse3 rmse4 rmse5 rmse6 rmse7 rmse8];
                        
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
    
end

save_name = strcat('output\SIM_UnknownGroup_K',num2str(K0),'_parchoice',num2str(parchoice),'.mat');
save(save_name);



%% RMSE
[reshape(mean(cell2mat(dataholder{1,5})),6,3) ...
    reshape(mean(cell2mat(dataholder{2,5})),6,3)...
    reshape(mean(cell2mat(dataholder{3,5})),6,3)]





%% Information Criterion
% change jj to get results for different sample sizes
jj = 1;
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
