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
    spec = strcat(spec,'_NoYlag_');
else
    spec = strcat(spec,'_Ylag_');
end

fprintf(strcat('Dependent Variable:\t',varname{1,par.y_idx},'\n'));
fprintf(strcat('Policy Variable:\t',varname{1,par.x_idx},'\n'));
fprintf(strcat('Shock Variable:\t',varname{1,par.z_idx},'\n'));
fprintf(strcat('Control Variable:\t',strjoin(varname(1,par.c_idx)),'\n'));

%% Now you have three options: individual LP, panel LP or GLP
H          = par.horizon;
LineColors = [.0  .2  .4];
BandColors = [.7  .7  .7];

%% 1) panel LP-IV
panOut = panel_LP(tmp, FE);
Ub_PAN = reshape(panOut.IRUb,1,par.horizon);
Lb_PAN = reshape(panOut.IRLb,1,par.horizon);
IR_PAN = reshape(panOut.IR,1,par.horizon);

% plotting
figure;
hold on;
fill([1:H, fliplr(1:H)],...
    [Ub_PAN fliplr(Lb_PAN)],...
    BandColors,'EdgeColor','none');
plot(1:H, IR_PAN,'LineWidth',1.2,'color',LineColors);
yline(0,'k','LineWidth',.7);
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;
saveas(gcf,strcat('./output/EMP/EMP_',spec,'PAN.png'));


%% 2) Individual LP-IV
indOut = ind_LP(tmp);
Ub_IND = reshape(indOut.IRUb,par.N,par.horizon);
Lb_IND = reshape(indOut.IRLb,par.N,par.horizon);
IR_IND = reshape(indOut.IR,par.N,par.horizon);

% Average Instrument Strength
fprintf('Average First-stage F stat: %f \n', mean(indOut.F))

% All IR plots
figure;
hold on;
plot(1:H, IR_IND,'LineWidth',1.2,'color',LineColors);
yline(0,'k','LineWidth',.7);
xlim([1 H]); axis tight
set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
    'FontSize',8,'Layer','top')
hold off;
saveas(gcf,strcat('./output/EMP/EMP_',spec,'IND.png'));

% Selected IR plots
for id = [250,357]
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
    saveas(gcf,strcat('./output/EMP/EMP_',spec,'IND_',MSA{1,id},'.png'));
end

%% 3) Group LP
Gmax   = 8;
nInit  = 50;
bInit  = indOut.b;
weight = repmat(mean(indOut.v_hac,3),1,1,par.N,1);
inference = 1;
[Gr_EST, GIRF, GSE, OBJ, IC] = GLP(tmp, Gmax, nInit, bInit, weight, FE, inference);

save(strcat('./output/EMP/EMP_',spec,'.mat'));

% information criterion
figure;
plot(IC,'b-s','LineWidth',2,'MarkerSize',5,...
    'MarkerEdgeColor','blue',...
    'MarkerFaceColor','blue');
xlabel('Number of Groups');
saveas(gcf,strcat('./output/EMP/EMP_',spec,'IC.png'));

% relabel the groups and plot them
Group_relabel = nan(par.N,Gmax);
for Ghat = 1:Gmax
    girf   = squeeze(GIRF{1,Ghat});
    gse    = squeeze(GSE{1,Ghat});
    Ub_GLP = girf + 1.96*gse;
    Lb_GLP = girf - 1.96*gse;

    % order ir by positiveness
    [~,ord] = sort(mean(girf,2),'descend');
    if Ghat>=2 && Ghat<=4
        figure;
        for g = 1:Ghat
            subplot(ceil(Ghat/2),2,g);
            k = ord(g);
            hold on;
            % bands
            fill([1:H, fliplr(1:H)],...
                [Ub_GLP(k,:) fliplr(Lb_GLP(k,:))],...
                BandColors,'EdgeColor','none');
            % IR
            plot(1:H, girf(k,:),'LineWidth',1.2,'color',LineColors);

            yline(0,'k','LineWidth',.7);
            xlabel(strcat({'Group'},{' '},num2str(g)));
            xlim([1 H]); axis tight
            set(gca,'XTick',[1 6 12 18 24],'XTickLabel',cellstr(num2str([1 6 12 18 24]')),...
                'FontSize',8,'Layer','top')
            hold off
        end
        saveas(gcf,strcat('./output/EMP/EMP_',spec,'GIRF_G',num2str(Ghat),'.png'));
    end
    % store ordered group classification
    gr_tmp = zeros(par.N,1);
    for g = 1:Ghat
        gr_tmp = gr_tmp +(Gr_EST(:,Ghat) == ord(g) )*g;
    end
    Group_relabel(:,Ghat) = gr_tmp; % store it for later output
end
writematrix(Group_relabel,strcat('./output/EMP/EMP_',spec,'Gr_re.csv'));