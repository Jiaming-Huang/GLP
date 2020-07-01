% Group Local Projection
% Version: 2020/07/01
% Empirical application - Alternative: FAVAR
% Jiaming Huang

close all; clear all; clc;
addpath('./routines');
addpath('./fred');
addpath('./svar');


%% Load Data
load EMP_data.mat

% get variables for VAR
varid = [8 16 19 22 24]; 
zid   = [13]; 
vars  = data(1:par.Tfull,varid);
proxy = data(1:par.Tfull,zid);

% get all the information variables
HPINFL = reshape(data(:,17),par.Tfull,par.N);

% merge two dataset
dateshp  = datetime(unique(date));
infoVars = HPINFL(2:end,:);
infoNames= MSA(1,:);

figure;plot(infoVars);

%% Factor Estimation: Onatski (2010)
DEMEAN=2;
jj=4;   % 4 is the Onatski criterion; % 5 is pre-specified number of factors
kmax=8;

[infoVarsrmo,~]=remove_outliers(infoVars);
[ehat,Fhat,lamhat,ve2,x2,mut,sdt] = factors_em(infoVarsrmo,kmax,jj,DEMEAN);

% Onatski (2010) chooses only a single factor
% Bai & Ng (2002) pc2 chooses 8 factors
nfac = size(Fhat,2);

% Explore the performance of the factor model
[R2,mR2,mR2_F,R2_T,t10_s,t10_mR2] = mrsq(Fhat,lamhat,ve2,infoNames);

% R2 for each individual series
figure;plot(R2)

fprintf('Mean R2 for Housing Inflation is %f \n\n',mean(R2(1:size(MSA,2))));
fprintf('Top 10 series explained by the factor is \n');
fprintf('%s \n',t10_s{:});


%% Put the factor into a standard VAR

%% Model Specification
modelSpec.start_date        = ['1990-01-01'];
modelSpec.end_date          = ['2007-12-01'];
modelSpec.DET               = [1 0 0];      % contant/trend/quadratic trend
modelSpec.p                 = 4;           % number of lags
modelSpec.irhor             = 24;           % maximal horizons
for i = 1:nfac
    facname(1,i) = {strcat('FAC',num2str(i))};
end
modelSpec.select_variables  = [varname(varid) facname];

% shock 
modelSpec.scale             = 1;
modelSpec.NWlags            = 8;

% inference
modelSpec.cLevel            = 95;     % confidence interval level
modelSpec.nBoot             = 1000;        % # of bootstrap
modelSpec.bootMethod        = 2;           % 1 - wild bootstrap;
                                           % 2 - Delta Method
                                           % 3 - Jentsch Lunsford MBB

%% Prepare the data
SVAR        = modelSpec;   % inherit all the parameters in the modelSpec
SVAR.n      = size(modelSpec.select_variables,2);

% sampling periods
sDate       = modelSpec.start_date;
eDate       = modelSpec.end_date;
ismpl       = dateshp>=datetime(sDate) & dateshp<=datetime(eDate);

% main variables
y           = [vars [nan(1,nfac);Fhat]];
y           = y(ismpl,:);       % Now T by K data is ready
z           = proxy(ismpl,:);
dateVAR     = dateshp(ismpl);

% check if there is any missing values, if so, modify sampling periods
naDetect = any(isnan([y z]),2);
if sum(naDetect)>0
    y = y(~naDetect,:);
    z = z(~naDetect,:);
    dateVAR = dateVAR(~naDetect);
    dmod = strcat('Time range has been modified from\t',datestr(dateVAR(1)),' to\t',datestr(dateVAR(end)),'.\n');
    fprintf('Your data contains missing values.\n');
    fprintf(dmod);
end


% determinants
T           = size(y,1);
DET         = [ones(T,1) [1:T]' [1:T].^2'];
DET         = DET(:,logical(modelSpec.DET));

% indicator for differenced variables
idiff       = [0 0 1 1 1 zeros(1,nfac)];

% pack variables
SVAR.vars  = y;
SVAR.proxy = z;
SVAR.DET   = DET;
SVAR.dates = dateVAR;
SVAR.idiff = idiff;

%% Step 2: Compute IRFs & Inference
SVAR       = doProxySVAR(SVAR); 
SVARIVci   = doProxySVARci(SVAR);

SVAR.irsH  = SVARIVci.irsH;
SVAR.irsL  = SVARIVci.irsL;
SVAR.Waldstat  = SVARIVci.Waldstat;



%% PART III: Plotting
irhor     = SVAR.irhor;
n         = SVAR.n;
SVAR.select_variables{1,3} = 'INDPRO';
SVAR.select_variables{1,4} = 'REALLN';
SVAR.select_variables{1,5} = 'INFL (PCE)';

figure;
for i = 1:n-nfac
    subplot(2,3,i);
    hold on;
        
    % zero line
    plot(0:irhor-1,zeros(size(0:irhor-1)),'k')    

    % irf
    plot(0:irhor-1,SVAR.irs(:,i),'k-','LineWidth',2);
    % cbands
    plot(0:irhor-1,SVAR.irsH(:,i),'r--');
    plot(0:irhor-1,SVAR.irsL(:,i),'r--');
    
    title(SVAR.select_variables(i));
end
    
    
%% Recover IRFs for information variables
irf_fac = SVAR.irs(:,6:end);
irf_fac_h = SVAR.irsH(:,6:end);
irf_fac_l = SVAR.irsL(:,6:end);

lamhat = lamhat.*sdt(1,:)';
irf_x = lamhat*irf_fac';
irf_x_h = lamhat*irf_fac_h';
irf_x_l = lamhat*irf_fac_l';

% interested in HP Inflation
figure;
plot(irf_x(1:size(MSA,2),:)','k-')

figure;
plot(cumsum(irf_x(1:size(MSA,2),:),2)','k-')


% Los Angeles
id = 211;
figure;plot(cumsum(irf_x(id,:)),'k-');
hold on; plot(cumsum(irf_x_h(id,:)),'r--');
plot(cumsum(irf_x_l(id,:)),'r--');
yline(0,'k--');

% Victoria
id = 357;
figure;plot(cumsum(irf_x(id,:)),'k-');
hold on; plot(cumsum(irf_x_h(id,:)),'r--');
plot(cumsum(irf_x_l(id,:)),'r--');
yline(0,'k--');