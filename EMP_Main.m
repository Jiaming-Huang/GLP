% Group Local Projection
% Version: 2021/03/08
% Author: Jiaming Huang
% There are four possibilities (change line 33 and 38)
%       - FE: lagged y - potential dynamic panel bias (but T is relatively large)
%       - FE: no lagged y - then why do FE? all rhs variables are aggregates
%       - RE: no lagged y - least efficient specification
%       - RE: lagged y - potential violation of RE assumption


close all; clear all; clc;
addpath('./routines');

%% Empirical Application
%% Load Data
% dum         = importdata('..\data\empirical_main.csv');
% data        = dum.data;
% date        = dum.textdata(2:end,1);
% MSA         = dum.textdata(2:end,2);
% varname     = strsplit(dum.textdata{1,1},','); varname = varname(1,3:end);
% par.N       = length(unique(MSA));
% par.Tfull   = size(data,1)/par.N;
% par.date    = date(1:par.Tfull,1);
% MSA         = reshape(MSA,par.Tfull,par.N);
% clear dum
% save('./output/EMP_data.mat');

load('./output/EMP_data.mat');

%% Model Specification
% 1. feed the index of your x, y and z
% 2. specify the number of lags
FE = 1;
par.y_idx = 18;         % housing inflation, log difference *100
par.x_idx = 8;          % FFR - 8; GS1 - 9
par.w_idx = [16 20 23 25];    %  FRM30, GIP, INFL_PCE, reloan
par.z_idx = [13];       % FF4 - 12; info robust IV1 - 13; MPSign - 14; RR - 15; 
par.nylag = 4;
par.nxlag = 4;
par.nwlag = 4;
par.nzlag = 0;
par.horizon = 24;       % horizons
par.nwtrunc = 25;
par.start = '1975-01-01';
par.end   = '2007-12-01';
reg = prepare_GLP(data, par);

tmp = reg;
tmp.LHS = cumsum(tmp.LHS,2);  % recover level

%% Now you have three options: individual LP, panel LP or GLP
H          = par.horizon;
LineColors = [.0  .2  .4];          
BandColors = [.7  .7  .7];

%% 1) panel LP-IV
[b, se] = panel_LP(tmp, FE);

% plotting
figure; 
hold on;
ub = b+1.96*se;
lb = b-1.96*se;
fill([1:H, fliplr(1:H)],...
    [ub fliplr(lb)],...
    BandColors,'EdgeColor','none');
plot(1:H, b,'LineWidth',1.2,'color',LineColors);            
yline(0,'k','LineWidth',.7);  
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;



%% 2) Individual LP-IV
[b_id, se_id, F_id] = ind_LP(tmp);
mean(F_id)

figure;
hold on;
plot(1:H, b_id','LineWidth',1.2,'color',LineColors); 
yline(0,'k','LineWidth',.7);  
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;



figure; 
hold on;
id =211; % Los Angeles
ub = b_id(id,:)+1.96*se_id(id,:);
lb = b_id(id,:)-1.96*se_id(id,:);
fill([1:H, fliplr(1:H)],...
    [ub fliplr(lb)],...
    BandColors,'EdgeColor','none');
plot(1:H, b_id(id,:),'LineWidth',1.2,'color',LineColors);            
yline(0,'k','LineWidth',.7);  
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;


figure; 
hold on;
id =357; % Los Angeles
ub = b_id(id,:)+1.96*se_id(id,:);
lb = b_id(id,:)-1.96*se_id(id,:);
fill([1:H, fliplr(1:H)],...
    [ub fliplr(lb)],...
    BandColors,'EdgeColor','none');
plot(1:H, b_id(id,:),'LineWidth',1.2,'color',LineColors);            
yline(0,'k','LineWidth',.7);  
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;

%% 3) Group LP
Gmax   = 10;
nInit  = 100;
bInit  = b_id;
weight = se_id;
inference = 3;
[Gr_EST, GIRF, GSE, IC] = GroupLPIV(tmp, Gmax, nInit, bInit, weight, FE, inference);

save('./output/EMP/EMP_FE_Y.mat');


Group = emp_relabel_plot(FE, par, Gr_EST, GIRF, GSE);

%% Plot IC
figure;
plot(IC,'b-s','LineWidth',2,'MarkerSize',5,...
'MarkerEdgeColor','blue',...
'MarkerFaceColor','blue');xlabel('Number of Groups');

%% Exisitng Criteria: Ad-hoc panel LP
% get the rich and poor index from data/MSA_Feature.xlsx, there are four
% sheets
% rich MSAs
idx = rich;
idx = kron(idx,ones(reg.param.T,1));idx = logical(idx);
adhoc.y = reg.y(idx,:);
adhoc.x = reg.x(idx,:);
adhoc.z = reg.z(idx,:);
adhoc.LHS = cumsum(reg.LHS(idx,:),2);
adhoc.control = reg.control(idx,:);
adhoc.param.T = reg.param.T;
adhoc.param.N = sum(rich==1);

[b_ad, se_ad] = panel_LP(adhoc,0);

figure;
hold on;
ub = b_ad+1.96*se_ad;
lb = b_ad-1.96*se_ad;
fill([1:H, fliplr(1:H)],...
    [ub fliplr(lb)],...
    BandColors,'EdgeColor','none');
plot(1:H, b_ad,'LineWidth',1.2,'color',LineColors);            
yline(0,'k','LineWidth',.7);  
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;


% poor MSAs
idx = poorG3_1;
idx = kron(idx,ones(reg.param.T,1));idx = logical(idx);
adhoc.y = reg.y(idx,:);
adhoc.x = reg.x(idx,:);
adhoc.z = reg.z(idx,:);
adhoc.LHS = cumsum(reg.LHS(idx,:),2);
adhoc.control = reg.control(idx,:);
adhoc.param.T = reg.param.T;
adhoc.param.N = sum(poorG3_1);

[b_ad, se_ad] = panel_LP(adhoc,0);

figure;
hold on;
ub = b_ad+1.96*se_ad;
lb = b_ad-1.96*se_ad;
fill([1:H, fliplr(1:H)],...
    [ub fliplr(lb)],...
    BandColors,'EdgeColor','none');
plot(1:H, b_ad,'LineWidth',1.2,'color',LineColors);            
yline(0,'k','LineWidth',.7);  
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;

