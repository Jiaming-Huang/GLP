function reg = preEmpData(data, par)
% Prepare data for empirical application: given the index of variables, 
% extract them, prepare lags and drop missings
% --------------------------- INPUT --------------------------------
% data: data matrix of size NT by p (balanced)
% par : a struct with 1) y_idx, x_idx, c_idx, zx_idx, zc_idx
%                           the index of y, x, c, zx, and zc; 
%                     2) nylag, nxlag, nclag, nzlag
%                           # of lags as additional controls
%                           by default nzlag should be 0
%                     3) start, end
%                           starting & ending dates of form 'yyyy-mm-dd'
%                     4) N, Tfull, # of units & (full) time periods
%                     5) date, Tfull by 1 vector of dates
% --------------------------- OUTPUT --------------------------------
% reg: data struct with y, x, (and z), LHS, controls and params that can be fed
%         into GroupLP function

%% Step 1: Keep only sampling periods
idxStart = find(strcmp(par.date, par.start));
idxEnd   = find(strcmp(par.date, par.end));
keep    = zeros(par.Tfull,1); 
keep(idxStart:idxEnd,1) = 1;
T       = sum(keep);
date    = par.date(logical(keep),1);
keep    = kron(ones(par.N,1),keep);
data    = data(logical(keep),:);

%% Step 2: Extract variables y, x, c, zx, zc
y = data(:,par.y_idx);
x = data(:,par.x_idx);
c = data(:,par.c_idx);   % if we don't have controls, par.c_idx should be []
zx = data(:,par.z_idx);
K = size(x,2);
L = size(zx,2);     % L by 1 instrument

%% Step 3: Generate lags
if isempty(c) && isempty(zx)
    %% Simplest model: OLS + no external controls
    ylag = nan(size(y,1),size(y,2)*par.nylag);
    xlag = nan(size(x,1),size(x,2)*par.nxlag); 
    for i = 1:par.N
        ytmp = y(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        
        ylag(T*(i-1)+1:T*i,:) = lag(ytmp,par.nylag);
        xlag(T*(i-1)+1:T*i,:) = lag(xtmp,par.nxlag);
        ylead(T*(i-1)+1:T*i,:) = lag(ytmp,-par.horizon);
    end
    tmp = [y x ylead ylag xlag];
    keep  = sum(isnan(tmp),2)==0;
    reg.y = tmp(keep,1);
    reg.x = tmp(keep,2:1+K);
    reg.LHS = tmp(keep,2+K:1+K+par.horizon);
    reg.control = tmp(keep,1+K+par.horizon+1:end);
    
elseif isempty(c)
    %% LP-IV: but no external controls
    ylag = nan(size(y,1),size(y,2)*par.nylag);
    xlag = nan(size(x,1),size(x,2)*par.nxlag);
    zlag = nan(size(zx,1),size(zx,2)*par.nzlag);    
    for i = 1:par.N
        ytmp = y(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        ztmp = zx(T*(i-1)+1:T*i,:);
        
        ylag(T*(i-1)+1:T*i,:) = lag(ytmp,par.nylag);
        xlag(T*(i-1)+1:T*i,:) = lag(xtmp,par.nxlag);
        zlag(T*(i-1)+1:T*i,:) = lag(ztmp,par.nzlag);
        ylead(T*(i-1)+1:T*i,:) = lag(ytmp,-par.horizon);
    end
    tmp = [y x zx ylead ylag xlag zlag];
    keep    = sum(isnan(tmp),2)==0;
    reg.y   = tmp(keep,1);
    reg.x   = tmp(keep,2:1+K);
    reg.zx  = tmp(keep,2+K:1+K+L);
    reg.LHS = tmp(keep,1+K+L+1:1+K+L+par.horizon);
    reg.c   = tmp(keep,1+K+L+par.horizon+1:end);
    
elseif isempty(zx)
    %% OLS: with external controls
    ylag = nan(size(y,1),size(y,2)*par.nylag);
    xlag = nan(size(x,1),size(x,2)*par.nxlag);
    clag = nan(size(c,1),size(c,2)*par.nclag);
    
    for i = 1:par.N
        ytmp = y(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        ctmp = c(T*(i-1)+1:T*i,:);
        
        ylag(T*(i-1)+1:T*i,:) = lag(ytmp,par.nylag);
        xlag(T*(i-1)+1:T*i,:) = lag(xtmp,par.nxlag);
        clag(T*(i-1)+1:T*i,:) = lag(ctmp,par.nclag);
        ylead(T*(i-1)+1:T*i,:) = lag(ytmp,-par.horizon);
    end
    tmp = [y x ylead c ylag xlag clag];
    keep  = sum(isnan(tmp),2)==0;
    reg.y = tmp(keep,1);
    reg.x = tmp(keep,2:1+K);
    reg.LHS = tmp(keep,1+K+1:1+K+par.horizon);
    reg.c = tmp(keep,1+K+par.horizon+1:end);

else
    %% LP-IV: with external controls
    ylag = nan(size(y,1),size(y,2)*par.nylag);
    xlag = nan(size(x,1),size(x,2)*par.nxlag);
    clag = nan(size(c,1),size(c,2)*par.nclag);
    zlag = nan(size(zx,1),size(zx,2)*par.nzlag);
    ylead = nan(size(y,1),par.horizon);
    for i = 1:par.N
        ytmp = y(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        ctmp = c(T*(i-1)+1:T*i,:);
        ztmp = zx(T*(i-1)+1:T*i,:);
        
        ylag(T*(i-1)+1:T*i,:) = lag(ytmp,par.nylag);
        xlag(T*(i-1)+1:T*i,:) = lag(xtmp,par.nxlag);
        clag(T*(i-1)+1:T*i,:) = lag(ctmp,par.nclag);
        zlag(T*(i-1)+1:T*i,:) = lag(ztmp,par.nzlag);
        ylead(T*(i-1)+1:T*i,:) = lag(ytmp,-par.horizon);
    end
    tmp = [y x zx ylead c ylag xlag clag zlag];
    keep  = sum(isnan(tmp),2)==0;
    reg.y = tmp(keep,1);
    reg.x = tmp(keep,2:1+K);
    reg.zx = tmp(keep,1+K+1:1+K+L);
    reg.LHS = tmp(keep,1+K+L+1:1+K+L+par.horizon);
    reg.c = tmp(keep,1+K+L+par.horizon+1:end);
end


% params
param.T = sum(keep)/par.N;
param.N = par.N;
param.date = date(keep(1:T,1),1);
param.nwtrunc = par.nwtrunc;
reg.param = param;
end