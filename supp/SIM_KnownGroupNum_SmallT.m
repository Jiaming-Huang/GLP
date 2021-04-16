%% Simulation Study #4 : known group number & small T
% See Section 6.2 in the paper
% results are stored in the subfolder /output/appen_smallT/SIM_KnownGroup_...
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
dataholder = cell(2,2,9);

%% Parameter Setting
% EST
FE         = 1;
inference  = 3;
H          = 6;               % h=0 is added in data preparation by default

% SIM
nRep       = 1000;
burn       = 100;
Ngrid      = [100, 200, 300];
T          = 50;

DGPsetup.H   = H;
DGPsetup.burn= burn;

for G0 = 2:3
    for parchoice = 1:2
        par        = params(G0,parchoice);
        DGPsetup.par = par;
        %% True IR
        IR_true = zeros(G0,H+1);
        for k = 1 : G0
            IR_true(k,:) = par(2,k)* (par(1,k) .^ [0:H]);
        end
        %% initialization, creating temporary statistics holder
        GLP_Group       = cell(nRep,size(Ngrid,2));
        GLP_IRF         = cell(nRep,size(Ngrid,2));
        GLP_GSE         = cell(nRep,size(Ngrid,2));
        GLP_ac          = nan(nRep, size(Ngrid,2));
        GLP_rmse        = nan(nRep, size(Ngrid,2));
        GLP_Coverage    = cell(nRep,size(Ngrid,2));
        GLP_band_len    = cell(nRep,size(Ngrid,2));
        
        IGLP_IRF        = cell(nRep,size(Ngrid,2));
        IGLP_GSE        = cell(nRep,size(Ngrid,2));
        IGLP_Coverage   = cell(nRep,size(Ngrid,2));
        IGLP_rmse       = nan(nRep, size(Ngrid,2));    
        
        IND_rmse        = nan(nRep,size(Ngrid,2));
        PAN_rmse        = nan(nRep,size(Ngrid,2));
        
        for jj = 1:size(Ngrid,2)
            N   = Ngrid(jj);
            fprintf('Start working on grid [N=%d, T=%d] \n', N, T);
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
            
            %% create IRF_TRUE for computing RMSE
            IR_TRUE = nan(N,H+1);
            for i =1:N
                IR_TRUE(i,:) = IR_true(Gr0(i),:);
            end
            
            
            %% Simulation starts here
            tic
            for iRep = 1:nRep
                Sim = DGP(N,T,DGPsetup);
                
                %% Benchmark: Panel LP-IV
                [b,~]=panel_LP(Sim.reg, FE);
                PAN_rmse(iRep,jj) = sqrt(mean( mean( (b - IR_TRUE).^2)));
                
                %% Benchmark: Individual LP-IV
                [b,se,~]=ind_LP(Sim.reg);
                IND_rmse(iRep,jj) = sqrt(mean( mean( (b - IR_TRUE).^2)));
                
                %% GLP Estimation
                weight = se;
                [Gr, GIRF, GSE]   = GroupLPIV_Sim_Known_Group(Sim.reg, G0, IR_true, weight, FE, inference);
                [GLP_ac(iRep,jj), GLP_rmse(iRep,jj), Permutation, Ng] = eval_GroupLPIV([Gr0 Gr], IR_TRUE, GIRF);
                GIRF = GIRF(Permutation,:);
                GSE  = GSE(Permutation,:);
                GLP_Group{iRep,jj}   = [Gr0 Gr];
                GLP_IRF{iRep,jj}     = GIRF;
                GLP_GSE{iRep,jj}     = GSE;
                Ubands = GIRF+1.96*GSE;
                Lbands = GIRF-1.96*GSE;
                
                GLP_Coverage{iRep,jj} = (Ubands > IR_true) & (Lbands < IR_true);
                GLP_band_len{iRep,jj} = mean(1.96*se*2) - Ng/N *(1.96*GSE*2);
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
                IGLP_IRF{iRep,jj}     = IGIRF;
                IGLP_GSE{iRep,jj}     = IGSE;
                IUbands = IGIRF+1.96*IGSE;
                ILbands = IGIRF-1.96*IGSE;
                
                IGLP_Coverage{iRep,jj} = (IUbands > IR_true) & (ILbands < IR_true);
                IGLP_rmse(iRep,jj) = getRMSE(Gr0,IR_TRUE,IGIRF);

                
            end
            t1 = toc;
            fprintf('Grid finished. Time used: %f seconds.\n', t1)
        end
        
        dataholder{G0-1,parchoice,1} = [mean(GLP_ac);mean(GLP_rmse);mean(PAN_rmse);mean(IND_rmse);mean(IGLP_rmse)];
        dataholder{G0-1,parchoice,2} = GLP_band_len;
        dataholder{G0-1,parchoice,3} = GLP_Coverage;
        dataholder{G0-1,parchoice,4} = GLP_Group;
        dataholder{G0-1,parchoice,5} = GLP_IRF;
        dataholder{G0-1,parchoice,6} = GLP_GSE;
        dataholder{G0-1,parchoice,7} = IGLP_IRF;
        dataholder{G0-1,parchoice,8} = IGLP_GSE;
        dataholder{G0-1,parchoice,9} = IGLP_Coverage;
        
        
    end
end
save_name = strcat('output\APPEN\SIM_SmallT_FE_noc.mat');
save(save_name);



%% for tables
% table appen smallT
% AC GLP PAN IND IGLP
[dataholder{1,1,1}';
    dataholder{2,1,1}';
    dataholder{1,2,1}';
    dataholder{2,2,1}']

% EFF
% for very small sample size, se can be funny because zero correl. of the
% instruments

tmp = cell2mat(dataholder{1,1,2});
tmp1 = tmp(:,1:7); tmp2 = tmp(:,8:14); tmp3 = tmp(:,15:end);
[mean(mean(tmp1(tmp1(:,1)<10,:)));
    mean(mean(tmp2(tmp2(:,1)<10,:)));
    mean(mean(tmp3(tmp3(:,1)<10,:)));]

tmp = cell2mat(dataholder{2,1,2});
tmp1 = tmp(:,1:7); tmp2 = tmp(:,8:14); tmp3 = tmp(:,15:end);
[mean(mean(tmp1(tmp1(:,1)<10,:)));
    mean(mean(tmp2(tmp2(:,1)<10,:)));
    mean(mean(tmp3(tmp3(:,1)<10,:)));]


tmp = cell2mat(dataholder{1,2,2});
tmp1 = tmp(:,1:7); tmp2 = tmp(:,8:14); tmp3 = tmp(:,15:end);
[mean(mean(tmp1(tmp1(:,1)<10,:)));
    mean(mean(tmp2(tmp2(:,1)<10,:)));
    mean(mean(tmp3(tmp3(:,1)<10,:)));]

tmp = cell2mat(dataholder{2,2,2});
tmp1 = tmp(:,1:7); tmp2 = tmp(:,8:14); tmp3 = tmp(:,15:end);
[mean(mean(tmp1(tmp1(:,1)<10,:)));
    mean(mean(tmp2(tmp2(:,1)<10,:)));
    mean(mean(tmp3(tmp3(:,1)<10,:)));]

squeeze([mean(mean(reshape(cell2mat(dataholder{1,2}),nRep,H+1,size(Tgrid,2))));
    mean(mean(reshape(cell2mat(dataholder{2,2}),nRep,H+1,size(Tgrid,2))));
    mean(mean(reshape(cell2mat(dataholder{3,2}),nRep,H+1,size(Tgrid,2))))])


% coverage rate
[reshape(mean(cell2mat(dataholder{1,1,3})),H+1,3)';
    reshape(mean(cell2mat(dataholder{2,1,3})),H+1,3)';
    reshape(mean(cell2mat(dataholder{1,2,3})),H+1,3)';
    reshape(mean(cell2mat(dataholder{2,2,3})),H+1,3)';]

[reshape(mean(cell2mat(dataholder{1,1,9})),H+1,3)';
    reshape(mean(cell2mat(dataholder{2,1,9})),H+1,3)';
    reshape(mean(cell2mat(dataholder{1,2,9})),H+1,3)';
    reshape(mean(cell2mat(dataholder{2,2,9})),H+1,3)';]

