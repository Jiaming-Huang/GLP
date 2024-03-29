%% Empirical Application
% See Section 7 in the paper
% Baseline Model (FE)
% y_it+h = mu_i + x_t*bet_gh + c_it*phi_ih + eps_it+h
% y_it+h    - housing inflation (log dif hp *100)
% xt        - FFR
% cit       - 4 lags of y, x, 30-year mortgage rates (FRM30), industrial production growth (GIP), PCE inflation (INFL_PCE), growth of real estate loans (reloan)
% zt        - info robust high-frequency MP shocks Miranda-Agrippino & Ricco

%% Housekeeping
% close all;
% clear; clc;
% addpath('./routines');

load('./data/EMP_data.mat');
rng(27);
%% Model Specification
% 1. feed the index of your x, y and z
% 2. specify the number of lags
FE = 1;
par.y_idx = 19;              % housing inflation, log difference *100
par.x_idx = 10;              % FFR - 10; GS1 - 11
par.c_idx = [18 21 24 26];    %  GIP, GREALLN, INFL_PCE
par.z_idx = 15;              % FF4 - 14; info robust IV1 - 15; MPSign - 16; RR - 17;
par.nylag = nylag;
par.nxlag = 4;
par.nclag = 4;
par.nzlag = 0;
par.horizon = 24;       % horizons
par.nwtrunc = 25;
par.start = '1975-01-01';
par.end   = '2007-12-01';
reg = preEmpData(data, par);

tmp = reg;
tmp.LHS = cumsum(tmp.LHS,2);  % recover level

spec = 'FE';
if par.nylag == 0
    spec = strcat(spec,'_NoYlag');
else
    spec = strcat(spec,'_Ylag');
end

fprintf(strcat('Dependent Variable:\t',varname{1,par.y_idx},'\n'));
fprintf(strcat('Policy Variable:\t',varname{1,par.x_idx},'\n'));
fprintf(strcat('Shock Variable:\t',varname{1,par.z_idx},'\n'));
fprintf(strcat('Control Variable:\t',strjoin(varname(1,par.c_idx)),'\n'));
%% Now you have three options: individual LP, panel LP or GLP
H          = par.horizon;
LineColors = [.0  .2  .4];
BandColors = [.7  .7  .7];


%% Exisitng Criteria: Ad-hoc panel LP
% get the rich and poor index from data/MSA_Feature_FE_Y.csv
MSA_Feature = importdata(strcat('./data/MSA_features_Adhoc_',spec,'.csv'));
% rich MSAs
idx = MSA_Feature.data(:,end-1);
idx = kron(idx,ones(tmp.param.T,1));
idx = logical(idx);
adhoc.y = tmp.y(idx,:);
adhoc.x = tmp.x(idx,:);
adhoc.zx = tmp.zx(idx,:);
adhoc.LHS = tmp.LHS(idx,:);
adhoc.c = tmp.c(idx,:);
adhoc.param.T = tmp.param.T;
adhoc.param.N = sum(MSA_Feature.data(:,end-1)==1);

richOut = panel_LP(adhoc, FE);
Ub_Rich = reshape(richOut.IRUb,1,par.horizon);
Lb_Rich = reshape(richOut.IRLb,1,par.horizon);
IR_Rich = reshape(richOut.IR,1,par.horizon);

% plotting
figure;
hold on;
fill([1:H, fliplr(1:H)],...
    [Ub_Rich fliplr(Lb_Rich)],...
    BandColors,'EdgeColor','none');
plot(1:H, IR_Rich,'LineWidth',1.2,'color',LineColors);
yline(0,'k','LineWidth',.7);
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;
saveas(gcf,strcat('./output/EMP/EMP_',spec,'_RICHp10.png'));



% poor MSAs
idx = MSA_Feature.data(:,end);
adhoc.param.N = sum(MSA_Feature.data(:,end));
idx = kron(idx,ones(tmp.param.T,1));
idx = logical(idx);
adhoc.y = tmp.y(idx,:);
adhoc.x = tmp.x(idx,:);
adhoc.zx = tmp.zx(idx,:);
adhoc.LHS = tmp.LHS(idx,:);
adhoc.c = tmp.c(idx,:);
adhoc.param.T = tmp.param.T;


poorOut = panel_LP(adhoc, FE);
Ub_Poor = reshape(poorOut.IRUb,1,par.horizon);
Lb_Poor = reshape(poorOut.IRLb,1,par.horizon);
IR_Poor = reshape(poorOut.IR,1,par.horizon);

% plotting
figure;
hold on;
fill([1:H, fliplr(1:H)],...
    [Ub_Poor fliplr(Lb_Poor)],...
    BandColors,'EdgeColor','none');
plot(1:H, IR_Poor,'LineWidth',1.2,'color',LineColors);
yline(0,'k','LineWidth',.7);
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;
saveas(gcf,strcat('./output/EMP/EMP_',spec,'_POORG1p10.png'));
