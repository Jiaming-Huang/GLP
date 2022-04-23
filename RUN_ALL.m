% This script includes all simulation exercises
clear;
close all;
clc;

addpath('./data');
addpath('./fred');
addpath('./output');
addpath('./routines');
addpath('./supp');
addpath('./svar');

rng(27);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                               MAIN DRAFT                             %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sec 6.1 Simululation Examples
% N            = 300;
% T            = 100;
% parchoice    = 1;
% H            = 6;
% DGPsetup.H   = H;
% DGPsetup.burn= 100;
% lColor = cell(3,2);
% lColor{1,1} = [0 0 0];
% lColor{1,2} = [0.5 0.5 0.5 0.3];
% lColor{2,1} = [1 0 0];
% lColor{2,2} = [0.5 0 0 0.3];
% lColor{3,1} = [0 1 0];
% lColor{3,2} = [0 0.5 0 0.3];
% for G0 = 2:3
%     par = params(G0,parchoice);
%     DGPsetup.par = par;
% 
%     % assign true membership
%     if G0 == 2
%         Ncut = N*[0.5 1];
%     elseif G0==3
%         Ncut = N*[0.3 0.6 1]; % for 3 groups
%     end
% 
%     id = 1:N;
%     Gr0  = ones(N,1)*G0;
%     for k=G0-1:-1:1
%         Gr0 = Gr0 - ( id <=Ncut(k) )' *1;
%     end
%     DGPsetup.G   = Gr0;
%     Sim = DGP(N,T,DGPsetup);
%     indOut = ind_LP(Sim.reg);
%     IR = squeeze(indOut.IR);
%     figure;
%     hold on;
%     for g = 1:G0
%         plot(IR(Gr0==g,:)','Color',lColor{g,2},'LineWidth',0.5);
%         % edges
%         plot(max(IR(Gr0==g,:)),'Color',lColor{g,1},'LineWidth',2.5);
%         plot(min(IR(Gr0==g,:)),'Color',lColor{g,1},'LineWidth',2.5);
%     end
%     saveas(gcf,strcat('./output/SIM_Ex_G',num2str(G0),'.png'));
% end


%% Sec 6.2 - known groups
% This section runs SIM_KnownG0.m for different parameter designs
% for G0 = 2:3
%     for parchoice  = 1:2
%         clc;clearvars -except G0 parchoice;
%         fprintf('Start working on G%d, Design %d\n', G0, parchoice)
%         SIM_KnownG0;
%     end
% end

%% Sec 6.3 - unknown groups
% This section runs SIM_UnknownG0.m for different parameter designs
% for parchoice  = 1:2
%     clc;clearvars -except parchoice;
%     fprintf('Start working on G3, Design %d\n', parchoice)
%     SIM_UnknownG0;
% end

%% Sec 6.4 No True Groups
% This section runs SIM_UnknownG0.m for DGP_NoGroup
% SIM_NoTrueGroup;

%% Sec 7.1 & 7.2 Empirical Application: GLP
% prepare data
% dum         = importdata('./data/empirical_main.csv');
% data        = dum.data;
% date        = dum.textdata(2:end,1);
% MSA         = dum.textdata(2:end,2);
% varname     = strsplit(dum.textdata{1,1},','); varname = varname(1,3:end);
% par.N       = length(unique(MSA));
% par.Tfull   = size(data,1)/par.N;
% par.date    = date(1:par.Tfull,1);
% MSA         = reshape(MSA,par.Tfull,par.N);
% clear dum
% save('./data/EMP_data.mat');
% baseline: FE & lags of dependent variables
% nylag = 4;
% close all;clc;clearvars -except nylag;
% EMP_Main;
% % adhoc grouping
% close all;clc;clearvars -except nylag;
% EMP_Adhoc;

%% Sec 7.3 Empirical Application: FAVAR
% baseline: onatski criterion
% jj=4;
% kmax=8;
% close all;clc;clearvars -except jj kmax;
% EMP_FAVAR;
% % user-specified number of factors, I choose 5 and 8 for illustration
% jj=5;
% kmax=5;
% close all;clc;clearvars -except jj kmax;
% EMP_FAVAR;
% kmax=8;
% close all;clc;clearvars -except jj kmax;
% EMP_FAVAR;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                        SUPPLEMENTAL MATERIALS                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sec S1.1 Different Sample Sizes - known groups
% This section runs SIM_KnownG0.m for different sample sizes
% Short panels: T=50, N=[100, 200, 300]
% Long panels: T=300, N=[1000, 1500, 2000]
% clear;
% close all;
% clc;
% SIM_SampSize_SmallT_KnownG0;
% clear;
% close all;
% clc;
% SIM_SampSize_LargeT_KnownG0;

%% Sec S1.2 Different Sample Sizes - unknown groups
% This section runs SIM_UnknownG0.m for different sample sizes
% Short panels: T=500, N=[100, 200]
% Long panels: T=[100,200,500] N=500
% clear;
% close all;
% clc;
% SIM_SampSize_SmallT_UnknownG0;
% clear;
% close all;
% clc;
% SIM_SampSize_LargeT_UnknownG0;

%% Sec S1.3 Different Weighting Matrix
% This section runs SIM_UnknownG0.m for different weighting matrices (2SLS, IV)
% for G0 = 2:3
%     for parchoice  = 1:2
%         clc;clearvars -except G0 parchoice;
%         fprintf('Start working on G%d, Design %d\n', G0, parchoice)
%         SIM_Weight_KnownG0;
%     end
% end

%% Sec S1.4 Different Inference Methods
% This section runs SIM_KnownG0_Inference.m
% for G0 = 2:3
%     for parchoice  = 1:2
%         clc;clearvars -except G0 parchoice;
%         fprintf('Start working on G%d, Design %d\n', G0, parchoice)
%         SIM_Inference_KnownG0;
%     end
% end

%% Sec S1.4 Different Specification
% This section runs SIM_UnknownG0.m for different specification
% In our baseline, we run FE (demeaning, and treat yit-1 as exogenous)
% Alternatively, we can estimate it by FD and use yit-2 as instrument (Anderson-Hsiao)
% for G0 = 2:3
%     for parchoice  = 1:2
%         clc;clearvars -except G0 parchoice;
%         fprintf('Start working on G%d, Design %d\n', G0, parchoice)
%         SIM_FDAH_KnownG0;
%     end
% end

%% Sec S1.5 Alternative Objective Function
% This section compares our GMM criterion and the fully pooled GMM
% for G0 = 2:3
%     for parchoice  = 1:2
%         clc;clearvars -except G0 parchoice;
%         fprintf('Start working on G%d, Design %d\n', G0, parchoice)
%         SIM_CompareGMMOBJ;
%     end
% end

%% Sec S1.6 Horizon-by-Horizon Grouping
% This section compares our the GLP with horizon-by-horizon grouping
% for G0 = 2:3
%     for parchoice  = 1:2
%         clc;clearvars -except G0 parchoice;
%         fprintf('Start working on G%d, Design %d\n', G0, parchoice)
%         SIM_HBH;
%     end
% end

%% Sec S2.4 Empirical Application: Dynamic panel bias
% nylag = 0;
% close all;clc;clearvars -except nylag;
% EMP_Main;
% close all;clc;clearvars -except nylag;
% EMP_Adhoc;

%% Sec S2.5 Empirical Application: Horizon-by-Horizon Grouping
% close all;
% clear; clc;
% EMP_HorizonByHorizon;
