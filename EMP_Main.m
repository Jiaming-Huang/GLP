% Group Local Projection
% Version: 2020/07/01
% Author: Jiaming Huang

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
% save EMP_data.mat

load EMP_data.mat
%% Model Specification
% 1. feed the index of your x, y and z
% 2. specify the number of lags

par.y_idx = 17;         % housing inflation, log difference *100
par.x_idx = 8;          % FFR - 8; GS1 - 9
par.w_idx = [16 19 22 24];    %  FRM30, GIP, INFL_PCE, reloan
par.z_idx = [13];       % FF4 - 12; info robust IV1 - 13; MPSign - 14; RR - 15; 
par.nylag = 4;
par.nxlag = 4;
par.nwlag = 4;
par.nzlag = 4;
par.horizon = 24;       % horizons
par.start = '1975-01-01';
par.end   = '2007-12-01';
reg = prepare_GLP(data, par);

tmp = reg;
tmp.LHS = cumsum(tmp.LHS,2);  % recover level

%% Now you have three options: individual LP, panel LP or GLP

%% 1) Individual LP-IV
[b_id, se_id, F_id] = ind_LP(tmp);
plot(F_id);mean(F_id)
figure(1)
plot(b_id','k-','LineWidth',1)


figure;
id =211; % Los Angeles
plot(b_id(id,:),'k-','LineWidth',2)
hold on
plot( (b_id(id,:)+1.96*se_id(id,:))','r--','LineWidth',2)
plot( (b_id(id,:)-1.96*se_id(id,:))','r--','LineWidth',2)
yline(0,'k--');
hold off

figure;
id =357; % Victoria
plot(b_id(id,:),'k-','LineWidth',2)
hold on
plot( (b_id(id,:)+1.96*se_id(id,:))','r--','LineWidth',2)
plot( (b_id(id,:)-1.96*se_id(id,:))','r--','LineWidth',2)
yline(0,'k--');
hold off

%% 2) panel LP-IV
[b, e, se] = panel_LP(tmp, 0);
figure(3)
plot(b','k-','LineWidth',2);
hold on;
plot(b+1.96*se,'r--','LineWidth',2);
plot(b-1.96*se,'r--','LineWidth',2);
yline(0,'k--');
hold off;

%% 3) Group LP
Kmax   = 10;
ninit  = 500;
tsls   = 1;
FE     = 0;   % 1 if include fixed effects, robust 
Group  = zeros(par.N,Kmax);
GIRF = cell(Kmax,1);
GSE  = cell(Kmax,1);
Qpath  = cell(Kmax,1);
OBJ    = nan(Kmax,1);
for Kguess = 1:Kmax
    % estimate with cumulative ones, robust to housing inflation
    [g, girf, gse, q, ~, ~] = GroupLPIV(tmp, Kguess, ninit, tsls, FE, b_id); 
    Group(:,Kguess)= g;
    GIRF{Kguess,1} = girf;
    GSE{Kguess,1}  = gse;
    Qpath{Kguess,1}= q; OBJ(Kguess,1)=min(q);
end

save Final_MSA_RobustIV.mat

% plot it
K0=3;   % you can make different plots by changing K0
btmp = GIRF{K0,1};
setmp = GSE{K0,1};
% relabeling so that smaller group number is associated with more positive
% IRs
if K0 <=3
    Kgrid = [1:K0];
elseif K0 == 4  
    Kgrid = [3 1 4 2];
elseif K0 == 5
    Kgrid = [5 1 2 4 3];
end
figure;
for i = 1:K0
    subplot(ceil(K0/2),2,i);
    k = Kgrid(i);
    plot(btmp(k,:),'k-','LineWidth',2)
    hold on
    plot( (btmp(k,:)+1.96*setmp(k,:))','r--')
    plot( (btmp(k,:)-1.96*setmp(k,:))','r--')
    yline(0,'k--')
    hold off
    xlabel(strcat({'Group'},{' '},num2str(i)));
end


%% Information criteria - upper bound of Kguess
t = reg.param.T;N=par.N;
plot(OBJ + [1:Kmax]'*OBJ(Kmax)*log(N*t) *(N+t)/(N*t),'b-s','LineWidth',2,'MarkerSize',5,...
'MarkerEdgeColor','blue',...
'MarkerFaceColor','blue');xlabel('Number of Groups');

%% Exisitng Criteria: Ad-hoc panel LP
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

[b_ad, ~, se_ad] = panel_LP(adhoc,0);

figure(10)
plot(b_ad,'k-','LineWidth',2)
hold on
plot( b_ad+1.96*se_ad,'r--','LineWidth',2)
plot( b_ad-1.96*se_ad,'r--','LineWidth',2)
hold off
yline(0,'k--')

% poor MSAs
poor = [149, 106, 23, 225, 216, 291, 360, 128, 381, 234];
idx = zeros(N,1); idx(poor',1)=1;
idx = kron(idx,ones(reg.param.T,1));idx = logical(idx);
adhoc.y = reg.y(idx,:);
adhoc.x = reg.x(idx,:);
adhoc.z = reg.z(idx,:);
adhoc.LHS = cumsum(reg.LHS(idx,:),2);
adhoc.control = reg.control(idx,:);
adhoc.param.T = reg.param.T;
adhoc.param.N = length(poor);

[b_ad, ~, se_ad] = panel_LP(adhoc,0);

figure;
plot(b_ad,'k-','LineWidth',2)
hold on
plot( b_ad+1.96*se_ad,'r--','LineWidth',2)
plot( b_ad-1.96*se_ad,'r--','LineWidth',2)
hold off
yline(0,'k--')













