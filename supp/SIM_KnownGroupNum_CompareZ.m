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
close all;
clear; clc;

rng(27);
addpath('./output');
addpath('./routines');
dataholder = cell(3,12);

%% Parameter Setting
% DGP
G0         = 3;               % true number of groups
if G0 == 3
    par = [0.3 0.6 0.9];
else
    par = [0.4 0.8];
end

% EST
FE         = 1;
inference  = 3;

% SIM
nRep       = 1000;
burn       = 100;
Ngrid      = [100, 200, 300];
Tgrid      = [100, 200, 300];


DGPsetup.par = par;
DGPsetup.lambda = 1.3;
DGPsetup.burn= burn;

%% True IRF
IR_true = zeros(G0,1);
for k = 1 : G0
    IR_true(k,:) = par(k);
end

%% Simulation
for jj = 1:size(Ngrid,2)
    % initialization, creating temporary statistics holder
    GLP_Group       = cell(nRep,size(Tgrid,2));
    GLP_IRF         = cell(nRep,size(Tgrid,2));
    GLP_GSE         = cell(nRep,size(Tgrid,2));
    GLP_ac          = nan(nRep,size(Tgrid,2));
    GLP_rmse        = nan(nRep,size(Tgrid,2));
    GLP_band_len    = nan(nRep,size(Tgrid,2));
    GLP_Coverage    = cell(nRep,size(Tgrid,2));
    GLP_F           = nan(nRep,size(Tgrid,2));
    GLP_Group1      = cell(nRep,size(Tgrid,2));
    GLP_IRF1        = cell(nRep,size(Tgrid,2));
    GLP_GSE1        = cell(nRep,size(Tgrid,2));
    GLP_ac1         = nan(nRep, size(Tgrid,2));
    GLP_rmse1       = nan(nRep, size(Tgrid,2));
    GLP_band_len1   = nan(nRep,size(Tgrid,2));
    GLP_Coverage1   = cell(nRep,size(Tgrid,2));
    GLP_F1          = nan(nRep,size(Tgrid,2));  
    
    N   = Ngrid(jj);
    
    % assign membership
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
    
    % create IRF_TRUE for computing RMSE
    IR_TRUE = nan(N,1);
    for i =1:N
        IR_TRUE(i,:) = IR_true(Gr0(i),:);
    end
    
    for tt = 1:size(Tgrid,2)
        T   = Tgrid(tt);
        fprintf('Start working on grid [N=%d, T=%d] \n', N, T)
        %% Simulation starts here
        tic
        for iRep = 1:nRep
            Sim = DGP_CompareZ1(N,T,DGPsetup);
            
            %% GLP Estimation - external IV
            [~,se,F]=ind_LP(Sim.reg);
            weight = se;
            [Gr, GIRF, GSE]   = GroupLPIV_Sim_Known_Group(Sim.reg, G0, IR_true, weight, FE, inference);
            [GLP_ac(iRep,tt), GLP_rmse(iRep,tt), Permutation, Ng] = eval_GroupLPIV([Gr0 Gr], IR_TRUE, GIRF);
            GIRF = GIRF(Permutation,:);
            GSE  = GSE(Permutation,:);
            GLP_Group{iRep,tt}   = [Gr0 Gr];
            GLP_IRF{iRep,tt}     = GIRF;
            GLP_GSE{iRep,tt}     = GSE;
            Ubands = GIRF+1.96*GSE;
            Lbands = GIRF-1.96*GSE;
            
            GLP_Coverage{iRep,tt} = (Ubands > IR_true) & (Lbands < IR_true);
            GLP_band_len(iRep,tt) = Ng/N *(1.96*GSE*2) / mean(1.96*se*2);
            GLP_F(iRep,tt) = mean(F);
            
            %% GLP Estimation - Arellano-Bond
            [~,se1,F1]=ind_LP(Sim.regAH);
            weight = se1;
            [Gr1, GIRF1, GSE1]   = GroupLPIV_Sim_Known_Group(Sim.regAH, G0, IR_true, weight, FE, inference);
            [GLP_ac1(iRep,tt), GLP_rmse1(iRep,tt), Permutation1, Ng1] = eval_GroupLPIV([Gr0 Gr1], IR_TRUE, GIRF1);
            GIRF1 = GIRF1(Permutation1,:);
            GSE1  = GSE1(Permutation1,:);
            GLP_Group1{iRep,tt}   = [Gr0 Gr1];
            GLP_IRF1{iRep,tt}     = GIRF1;
            GLP_GSE1{iRep,tt}     = GSE1;
            Ubands1 = GIRF1+1.96*GSE1;
            Lbands1 = GIRF1-1.96*GSE1;
            
            GLP_Coverage1{iRep,tt} = (Ubands1 > IR_true) & (Lbands1 < IR_true);
            GLP_band_len1(iRep,tt) = Ng1/N *(1.96*GSE1*2) / mean(1.96*se1*2);
            GLP_F1(iRep,tt) = mean(F1);

        end
        t1 = toc;
        fprintf('Grid finished. Time used: %f seconds.\n', t1)
    end
    
    %     etmp = reshape(cell2mat(GLP_band_len),nRep,H+1,size(Tgrid,2));
    dataholder{jj,1} = [mean(GLP_ac);mean(GLP_ac1);mean(GLP_rmse);mean(GLP_rmse1);];
    dataholder{jj,2} = GLP_band_len;
    dataholder{jj,3} = GLP_Coverage;
    dataholder{jj,4} = GLP_Group;
    dataholder{jj,5} = GLP_IRF;
    dataholder{jj,6} = GLP_GSE;
    dataholder{jj,7} = GLP_band_len1;
    dataholder{jj,8} = GLP_Coverage1;
    dataholder{jj,9} = GLP_Group1;
    dataholder{jj,10} = GLP_IRF1;
    dataholder{jj,11} = GLP_GSE1;
    dataholder{jj,12} = [GLP_F GLP_F1];
end

save_name = strcat('output\APPEN\SIM_CompareZ1_G',num2str(G0),'.mat');
save(save_name);



%% for tables
% F statistics
[reshape(mean(dataholder{1,12}),3,2);
    reshape(mean(dataholder{2,12}),3,2);
    reshape(mean(dataholder{3,12}),3,2);]
    

% AC RMSE
[dataholder{1,1}';
    dataholder{2,1}';
    dataholder{3,1}']

% EFF
[[mean(dataholder{1,2})';
    mean(dataholder{2,2})';
    mean(dataholder{3,2})';]   [mean(dataholder{1,7})';
    mean(dataholder{2,7})';
    mean(dataholder{3,7})';]]

% coverage rates
[[reshape(mean(cell2mat(dataholder{1,3})),1,3)';
    reshape(mean(cell2mat(dataholder{2,3})),1,3)';
    reshape(mean(cell2mat(dataholder{3,3})),1,3)'] [reshape(mean(cell2mat(dataholder{1,8})),1,3)';
    reshape(mean(cell2mat(dataholder{2,8})),1,3)';
    reshape(mean(cell2mat(dataholder{3,8})),1,3)']]
