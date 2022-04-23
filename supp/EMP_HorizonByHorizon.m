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

%% Load Data
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

load('EMP_data.mat');
%% Model Specification
% 1. feed the index of your x, y and z
% 2. specify the number of lags
FE = 1;
par.y_idx = 19;              % housing inflation, log difference *100
par.x_idx = 10;              % FFR - 10; GS1 - 11
par.c_idx = [18 21 24 26];    % FRM30 GIP, GREALLN, INFL_PCE
par.z_idx = 15;              % FF4 - 14; info robust IV1 - 15; MPSign - 16; RR - 17;
par.nylag = 4;
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

fprintf(strcat('Dependent Variable:\t',varname{1,par.y_idx},'\n'));
fprintf(strcat('Policy Variable:\t',varname{1,par.x_idx},'\n'));
fprintf(strcat('Shock Variable:\t',varname{1,par.z_idx},'\n'));
fprintf(strcat('Control Variable:\t',strjoin(varname(1,par.c_idx)),'\n'));

%% Now you have three options: individual LP, panel LP or GLP
H          = par.horizon;
LineColors = [.0  .2  .4];
BandColors = [.7  .7  .7];

%% Horizon-by-horizon Group LP
% initial guess
indOut = ind_LP(tmp);
weight = repmat(mean(indOut.v_hac,3),1,1,par.N,1);
Gmax   = 10;
nInit  = 50;
inference = 1;
Gr_EST = cell(1,H);
GIRF   = cell(1,H);
GSE    = cell(1,H);
OBJ    = nan(Gmax,H);
IC     = nan(Gmax,H);

for h = 1:H
    tmp_h     = tmp;
    tmp_h.LHS = tmp_h.LHS(:,h);
    bInit     = indOut.b(:,:,:,h);
    [Gr_EST{1,h}, GIRF{1,h}, GSE{1,h}, OBJ(:,h), IC(:,h)] = GLP(tmp_h, Gmax, nInit, bInit, weight(:,:,:,h), FE, inference);
end

save(strcat('.\output\EMP\EMP_HBH.mat'));

% information criterion
figure;
plot(IC,'b-s','LineWidth',1,'MarkerSize',5,...
    'MarkerEdgeColor','blue',...
    'MarkerFaceColor','blue');
xlabel('Number of Groups');
saveas(gcf,strcat('.\output\EMP\EMP_HBH_IC.png'));

% choose Ghat = 3 as an example
Ghat = 3;
girf = nan(Ghat,H);
gse  = nan(Ghat,H);
gr_est = nan(par.N,H);
for h = 1:H
    ir_tmp = GIRF{1,h};
    se_tmp = GSE{1,h};
    gr_tmp = Gr_EST{1,h};
    girf(:,h)= squeeze(ir_tmp{1,Ghat});
    gse(:,h)= squeeze(se_tmp{1,Ghat});
    gr_est(:,h) = gr_tmp(:,Ghat);
end

% Re-order the IRs for each horizon
Gr_relabel   = nan(par.N,H);
GIRF_relabel = nan(Ghat,H);
GSE_relabel  = nan(Ghat,H);
for h = 1:H
    [~, ord] = sort(girf(:,h),'descend');
    gr_tmp = zeros(par.N,1);
    for g = 1:Ghat
        gr_tmp = gr_tmp +(gr_est(:,h) == ord(g) )*g;
    end
    Gr_relabel(:,h)   = gr_tmp;
    GIRF_relabel(:,h) = girf(ord,h);
    GSE_relabel(:,h)  = gse(ord,h);
end

writematrix(Gr_relabel,strcat('output\EMP\EMP_HBH_Gr_re.csv'));


% Plot Group IRs
Ub = GIRF_relabel+1.96*GSE_relabel;
Lb = GIRF_relabel-1.96*GSE_relabel;
figure;
for g = 1:Ghat
    subplot(ceil(Ghat/2),2,g);
    hold on;
    % bands
    fill([1:H, fliplr(1:H)],...
        [Ub(g,:) fliplr(Lb(g,:))],...
        BandColors,'EdgeColor','none');
    % IR
    plot(1:H, GIRF_relabel(g,:),'LineWidth',1.2,'color',LineColors);

    yline(0,'k','LineWidth',.7);
    xlabel(strcat({'Group'},{' '},num2str(g)));
    xlim([1 H]); axis tight
    set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
        'FontSize',8,'Layer','top')
    hold off
end
saveas(gcf,strcat('.\output\EMP\EMP_HBH_GIRF_G',num2str(Ghat),'.png'));


% Plot funny individual IRs
ir_tmp = nan(1,H);
se_tmp = nan(1,H);
id = 63;
for h = 1:H
    ir_tmp(h) = GIRF_relabel(Gr_relabel(id,h),h);
    se_tmp(h) = GSE_relabel(Gr_relabel(id,h),h);
end
Ub_id = ir_tmp+1.96*se_tmp;
Lb_id = ir_tmp-1.96*se_tmp;

figure;
hold on;
fill([1:H, fliplr(1:H)],...
    [Ub_id fliplr(Lb_id)],...
    BandColors,'EdgeColor','none');
plot(1:H, ir_tmp,'LineWidth',1.2,'color',LineColors);
yline(0,'k','LineWidth',.7);
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;
saveas(gcf,strcat('.\output\EMP\EMP_HBH_Charleston_WV.png'));

% compare id = 63 with IND and GLP
load('./output/EMP/EMP_FE_Ylag_.mat')

% IND LP-IV
id = 63;
figure;
hold on;
fill([1:H, fliplr(1:H)],...
    [Ub_IND(id,:) fliplr(Lb_IND(id,:))],...
    BandColors,'EdgeColor','none');
plot(1:H, IR_IND(id,:),'LineWidth',1.2,'color',LineColors);
yline(0,'k','LineWidth',.7);
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;
saveas(gcf,strcat('.\output\EMP\EMP_IND_Charleston_WV.png'));

% GLP
gr_est = Gr_EST(id,Ghat);
girf   = squeeze(GIRF{1,Ghat});
gse    = squeeze(GSE{1,Ghat});
ub = girf+1.96*gse;
lb = girf-1.96*gse;
figure;
hold on;
fill([1:H, fliplr(1:H)],...
    [ub(gr_est,:) fliplr(lb(gr_est,:))],...
    BandColors,'EdgeColor','none');
plot(1:H, girf(gr_est,:),'LineWidth',1.2,'color',LineColors);
yline(0,'k','LineWidth',.7);
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;
saveas(gcf,strcat('.\output\EMP\EMP_GLP_Charleston_WV.png'));
