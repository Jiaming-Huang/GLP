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
dataholder = cell(3,9);

%% Parameter Setting
% DGP
G0         = 3;               % true number of groups
parchoice  = 1;
par        = params(G0,parchoice);

% EST
FE         = 1;
inference  = 3;
H          = 6;               % h=0 is added in data preparation by default

% SIM
nRep       = 1000;
burn       = 100;
Ngrid      = [100, 200, 300];
Tgrid      = [100, 200, 300];
% Ngrid      = [500, 1000, 1500, 2000];
% Tgrid      = [500];

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
    GLP_Group       = cell(nRep,size(Tgrid,2));
    GLP_IRF         = cell(nRep,size(Tgrid,2));
    GLP_GSE         = cell(nRep,size(Tgrid,2));
    GLP_ac          = nan(nRep, size(Tgrid,2));
    GLP_rmse        = nan(nRep, size(Tgrid,2));
    GLP_Coverage    = cell(nRep,size(Tgrid,2));
    GLP_band_len    = cell(nRep,size(Tgrid,2));
    IGLP_IRF        = cell(nRep,size(Tgrid,2));
    IGLP_GSE        = cell(nRep,size(Tgrid,2));
    IGLP_Coverage   = cell(nRep,size(Tgrid,2));
    IGLP_rmse       = nan(nRep, size(Tgrid,2));
    IND_rmse        = nan(nRep,size(Tgrid,2));
    PAN_rmse        = nan(nRep,size(Tgrid,2));
    
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

            %% Benchmark: Panel LP-IV
            [b,~]=panel_LP(Sim.reg, FE);
            PAN_rmse(iRep,tt) = sqrt(mean( mean( (b - IR_TRUE).^2)));
            
            %% Benchmark: Individual LP-IV
            [b,se,~]=ind_LP(Sim.reg);
            IND_rmse(iRep,tt) = sqrt(mean( mean( (b - IR_TRUE).^2)));
            
            %% GLP Estimation
            weight = se;
            [Gr, GIRF, GSE]   = GroupLPIV_Sim_Known_Group(Sim.reg, G0, IR_true, weight, FE, 3);
            [GLP_ac(iRep,tt), GLP_rmse(iRep,tt), Permutation, Ng] = eval_GroupLPIV([Gr0 Gr], IR_TRUE, GIRF);
            GIRF = GIRF(Permutation,:);
            GSE  = GSE(Permutation,:);
            GLP_Group{iRep,tt}   = [Gr0 Gr];
            GLP_IRF{iRep,tt}     = GIRF;
            GLP_GSE{iRep,tt}     = GSE;
            Ubands = GIRF+1.96*GSE;
            Lbands = GIRF-1.96*GSE;
            
            GLP_Coverage{iRep,tt} = (Ubands > IR_true) & (Lbands < IR_true);
            GLP_band_len{iRep,tt} = mean(1.96*se*2) - Ng/N *(1.96*GSE*2);            
%             figure;
%             for ip = 1:G0
%                 subplot(2,2,ip);
%                 plot(GIRF(ip,:),'k-');
%                 hold on;
%                 plot(IR_true(ip,:),'b-');
%                 plot(GIRF(ip,:)+1.96*GSE(ip,:),'r--');
%                 plot(GIRF(ip,:)-1.96*GSE(ip,:),'r--');
%                 hold off
%             end
            
            

            %% IGLP - Infeasible GLP (known Groups)
            [IGIRF, IGSE] = GroupLPIV_Infeasible(Sim.reg, Gr0, FE);
            IGLP_IRF{iRep,tt}     = IGIRF;
            IGLP_GSE{iRep,tt}     = IGSE;
            IUbands = IGIRF+1.96*IGSE;
            ILbands = IGIRF-1.96*IGSE;
            
            IGLP_Coverage{iRep,tt} = (IUbands > IR_true) & (ILbands < IR_true);
            IGLP_rmse(iRep,tt) = getRMSE(Gr0,IR_TRUE,IGIRF);
            %             fprintf('Iteration %d: \n', iRep)
        end
        t1 = toc;
        fprintf('Grid finished. Time used: %f seconds.\n', t1)
    end
    
%     etmp = reshape(cell2mat(GLP_band_len),nRep,H+1,size(Tgrid,2));
    dataholder{jj,1} = [mean(GLP_ac);mean(GLP_rmse);mean(PAN_rmse);mean(IND_rmse);mean(IGLP_rmse)];
    dataholder{jj,2} = GLP_band_len;
    dataholder{jj,3} = GLP_Coverage;
    dataholder{jj,4} = GLP_Group;
    dataholder{jj,5} = GLP_IRF;
    dataholder{jj,6} = GLP_GSE;
    dataholder{jj,7} = IGLP_IRF;
    dataholder{jj,8} = IGLP_GSE;
    dataholder{jj,9} = IGLP_Coverage;
    
end

save_name = strcat('output\SIM_KnownGroup_G',num2str(G0),'_param',num2str(parchoice),'FE_noc_jointinf.mat');
save(save_name);



%% for tables
% table 2
% AC GLP PAN IND IGLP
[dataholder{1,1}';
    dataholder{2,1}';
    dataholder{3,1}']
% EFF
[squeeze(mean(mean(reshape(cell2mat(dataholder{1,2}),nRep,H+1,size(Tgrid,2)))));
    squeeze(mean(mean(reshape(cell2mat(dataholder{2,2}),nRep,H+1,size(Tgrid,2)))));
    squeeze(mean(mean(reshape(cell2mat(dataholder{3,2}),nRep,H+1,size(Tgrid,2)))))]
    

% table 3
[reshape(mean(cell2mat(dataholder{1,3})),H+1,3)';
    reshape(mean(cell2mat(dataholder{2,3})),H+1,3)';
    reshape(mean(cell2mat(dataholder{3,3})),H+1,3)']

% Coverage rate: IGLP
[reshape(mean(cell2mat(dataholder{1,9})),H+1,3)';
    reshape(mean(cell2mat(dataholder{2,9})),H+1,3)';
    reshape(mean(cell2mat(dataholder{3,9})),H+1,3)']
