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
dataholder = cell(3,5);

%% Parameter Setting
% DGP
K0         = 3;               % true number of groups
parchoice  = 2;
par        = params(K0,parchoice);
% EST
TSLS       = 1; 
FE         = 1;
H          = 6;               % h=0 is added in data preparation by default
% SIM
nRep       = 1000;
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

H=6;
IRF = [1 2 3]'* (0.5 .^ [0:H]);

figure;plot([0:6],IRF','k-','LineWidth',2)
xlabel('horizon')
ylabel('impulse response')

%% Simulation
for jj = 1:size(Ngrid,2)
        % initialization, creating temporary statistics holder
        GLP_Group       = cell(nRep,size(Tgrid,2));
        GLP_IRF         = cell(nRep,size(Tgrid,2));
        GLP_ac          = nan(nRep, size(Tgrid,2));
        GLP_rmse        = nan(nRep, size(Tgrid,2));
        GLP_bias        = cell(nRep,size(Tgrid,2));
        GLP_Coverage    = cell(nRep,size(Tgrid,2));
        GLP_band_len    = cell(nRep,size(Tgrid,2));
        IGLP_IRF        = cell(nRep,size(Tgrid,2));
        IGLP_bias       = cell(nRep,size(Tgrid,2));  % under true grouping
        IGLP_rmse       = nan(nRep, size(Tgrid,2));
        IND_rmse        = nan(nRep,size(Tgrid,2));
        PAN_rmse        = nan(nRep,size(Tgrid,2));
        
        N   = Ngrid(jj);
        
        % assign membership
        if K0 == 2
            Ncut= N*[0.5 1];
        elseif K0==3
            Ncut= N*[0.3 0.6 1]; % for 3 groups
        end
        
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
        
        %% Simulation starts here
        for iRep = 1:nRep
            Sim = DGP(N,T,DGPsetup);
            
            %% GLP Estimation
            [Gr, GIRF, GSE, Qpath, ~]   = GroupLPIV_TrueIRF(Sim.reg, K0, IRF_true, TSLS, FE);
            [GLP_ac(iRep,tt), GLP_rmse(iRep,tt), GLP_bias{iRep,tt}, Permutation, Ng] = eval_GroupLPIV([G Gr], IRF_true, GIRF);
            GIRF = GIRF(Permutation,:);
            GSE  = GSE(Permutation,:);
            GLP_Group{iRep,tt}   = [G Gr];
            Ubands = GIRF+1.96*GSE;
            Lbands = GIRF-1.96*GSE;
            GLP_IRF{iRep,tt}     = GIRF;
            GLP_Coverage{iRep,tt} = (Ubands > IRF_true) & (Lbands < IRF_true);
            
            %% IGLP - Infeasible GLP (known Groups)
            [GIRF1, GSE1] = GroupLPIV_TrueGroup(Sim.reg, G, FE);
            [~, IGLP_rmse(iRep,tt), IGLP_bias{iRep,tt}, ~,~] = eval_GroupLPIV([G G], IRF_true, GIRF1);
            IGLP_IRF{iRep,tt}     = GIRF1;
            
            %% Benchmark: Individual LP and Panel LP
            [b,se,~]=ind_LP(Sim.reg);
            IND_rmse(iRep,tt) = sqrt(mean( mean( (b - IRF_TRUE).^2)));
            GLP_band_len{iRep,tt} = mean(1.96*se*2) - Ng/N *(1.96*GSE*2);
            [b,~,~]=panel_LP1(Sim.reg, FE);
            PAN_rmse(iRep,tt) = sqrt(mean( mean( (b - IRF_TRUE).^2)));
        end
        
    end
    
    btmp = reshape(cell2mat(GLP_bias),nRep*K0,H+1,size(Tgrid,2));
    etmp = reshape(cell2mat(GLP_band_len),nRep,H+1,size(Tgrid,2));
    dataholder{jj,1} = [mean(GLP_ac);mean(GLP_rmse);mean(PAN_rmse);mean(IND_rmse);mean(IGLP_rmse)];
    dataholder{jj,2} = squeeze(mean(etmp));
    dataholder{jj,3} = squeeze(mean(btmp));
    dataholder{jj,4} = GLP_Coverage;
    dataholder{jj,5} = GLP_IRF;
    dataholder{jj,6} = IGLP_IRF;
    
end

save_name = strcat('output\SIM_KnownGroup_K',num2str(K0),'_parchoice',num2str(parchoice),'.mat');
save(save_name);

%% for tables
% table 2
% AC GLP PAN IND IGLP
[dataholder{1,1}';
    dataholder{2,1}';
    dataholder{3,1}']  
% EFF
[mean(dataholder{1,2}) mean(dataholder{2,2}) mean(dataholder{3,2})]' 

% table 3
[reshape(mean(cell2mat(dataholder{1,4})),H+1,3)';
    reshape(mean(cell2mat(dataholder{2,4})),H+1,3)';
    reshape(mean(cell2mat(dataholder{3,4})),H+1,3)']
