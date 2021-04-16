function reg = prepare_GLP(data, par)
% given the index of variables, extract them, prepare lags and drop missings
% INPUT:
% -- data: data matrix of size NT by p
% -- par : a struct with 1) the index of y, x, (w and z); 
%                        2) # of lags
% by default, x and z should be of column 1!

% OUTPUT:
% -- reg: data struct with y, x, (and z), LHS, controls and params that can be fed
%         into GroupLP function



t_start = find(strcmp(par.date, par.start));
t_end   = find(strcmp(par.date, par.end));
keep    = zeros(par.Tfull,1); 
keep(t_start:t_end,1) = 1;
T       = sum(keep);
date    = par.date(logical(keep),1);
keep    = kron(ones(par.N,1),keep);
data    = data(logical(keep),:);

%% extract variables at time t
y = data(:,par.y_idx);
x = data(:,par.x_idx);
w = data(:,par.w_idx);   % if we don't have controls, par.w_idx should be []
z = data(:,par.z_idx);   % same for z_idx
L = size(par.z_idx,2);     % L by 1 instrument

if isempty(w) && isempty(z)
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
    reg.x = tmp(keep,2);
    reg.LHS = tmp(keep,2+1:2+par.horizon);
    reg.control = tmp(keep,2+par.horizon+1:end);
    reg.z = [];
    
elseif isempty(w)
    %% LP-IV: but no external controls
    ylag = nan(size(y,1),size(y,2)*par.nylag);
    xlag = nan(size(x,1),size(x,2)*par.nxlag);
    zlag = nan(size(z,1),size(z,2)*par.nzlag);    
    for i = 1:par.N
        ytmp = y(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        ztmp = z(T*(i-1)+1:T*i,:);
        
        ylag(T*(i-1)+1:T*i,:) = lag(ytmp,par.nylag);
        xlag(T*(i-1)+1:T*i,:) = lag(xtmp,par.nxlag);
        zlag(T*(i-1)+1:T*i,:) = lag(ztmp,par.nzlag);
        ylead(T*(i-1)+1:T*i,:) = lag(ytmp,-par.horizon);
    end
    tmp = [y x z ylead ylag xlag zlag];
    keep  = sum(isnan(tmp),2)==0;
    reg.y = tmp(keep,1);
    reg.x = tmp(keep,2);
    reg.z = tmp(keep,3:2+L);
    reg.LHS = tmp(keep,2+L+1:2+L+par.horizon);
    reg.control = tmp(keep,2+L+par.horizon+1:end);
    
elseif isempty(z)
    %% OLS: with external controls
    ylag = nan(size(y,1),size(y,2)*par.nylag);
    xlag = nan(size(x,1),size(x,2)*par.nxlag);
    wlag = nan(size(w,1),size(w,2)*par.nwlag);
    
    for i = 1:par.N
        ytmp = y(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        wtmp = w(T*(i-1)+1:T*i,:);
        
        ylag(T*(i-1)+1:T*i,:) = lag(ytmp,par.nylag);
        xlag(T*(i-1)+1:T*i,:) = lag(xtmp,par.nxlag);
        wlag(T*(i-1)+1:T*i,:) = lag(wtmp,par.nwlag);
        ylead(T*(i-1)+1:T*i,:) = lag(ytmp,-par.horizon);
    end
    tmp = [y x ylead w ylag xlag wlag];
    keep  = sum(isnan(tmp),2)==0;
    reg.y = tmp(keep,1);
    reg.x = tmp(keep,2);
    reg.LHS = tmp(keep,3:2+par.horizon);
    reg.control = tmp(keep,2+par.horizon+1:end);
    reg.z = [];
else
    %% LP-IV: with external controls
    ylag = nan(size(y,1),size(y,2)*par.nylag);
    xlag = nan(size(x,1),size(x,2)*par.nxlag);
    wlag = nan(size(w,1),size(w,2)*par.nwlag);
    zlag = nan(size(z,1),size(z,2)*par.nzlag);
    ylead = nan(size(y,1),par.horizon);
    for i = 1:par.N
        ytmp = y(T*(i-1)+1:T*i,:);
        xtmp = x(T*(i-1)+1:T*i,:);
        wtmp = w(T*(i-1)+1:T*i,:);
        ztmp = z(T*(i-1)+1:T*i,:);
        
        ylag(T*(i-1)+1:T*i,:) = lag(ytmp,par.nylag);
        xlag(T*(i-1)+1:T*i,:) = lag(xtmp,par.nxlag);
        wlag(T*(i-1)+1:T*i,:) = lag(wtmp,par.nwlag);
        zlag(T*(i-1)+1:T*i,:) = lag(ztmp,par.nzlag);
        ylead(T*(i-1)+1:T*i,:) = lag(ytmp,-par.horizon);
    end
    tmp = [y x z ylead w ylag xlag wlag zlag];
    keep  = sum(isnan(tmp),2)==0;
    reg.y = tmp(keep,1);
    reg.x = tmp(keep,2);
    reg.z = tmp(keep,3:2+L);
    reg.LHS = tmp(keep,2+L+1:2+L+par.horizon);
    reg.control = tmp(keep,2+L+par.horizon+1:end);
end


% params
param.T = sum(keep)/par.N;
param.N = par.N;
param.date = date(keep(1:T,1),1);
param.nwtrunc = par.nwtrunc;
reg.param = param;
end