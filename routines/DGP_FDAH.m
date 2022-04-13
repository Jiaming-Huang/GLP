function Sim = DGP_FDAH(N,T,DGPsetup)
% generate simulation data
% --------------------------- MODEL --------------------------------
% y_it = mu_i + phi_g*y_it-1 + beta_g*x_it + ep_it
% x_it = mu_i + pi*z_it + u_it
% mu_i ~ U(0,1)
% z_it is iid N(0,1)
% u_it, ep_it are bivarate normal, with Sig=[1 0.3; 0.3 1]

% --------------------------- INPUT --------------------------------
% N,T  - size of the panel (balanced)
% DGPsetup: a data struct of parameters, containing
%           1) phi  - persistence of impulse responses, group specific
%           2) beta - size of impulse responses, group specific
%           3) G    - N by 1 vector indicating group number
%           4) H    - impulse responses horizons
%           5) burn - number of burning periods

% --------------------------- OUTPUT --------------------------------
% Sim: a data struct ready for estimation, containing
%           1) S, Z, Y, X - raw data series for shocks S, instruments Z, 
%                               data Y and X
%           2) reg        - reshaped data for estimation
%           3) param      - N and T

%% Unpack Parameters
phi = DGPsetup.par(1,:);
bet = DGPsetup.par(2,:);
pi  = 0.7;
G   = DGPsetup.G;
H   = DGPsetup.H;
burn= DGPsetup.burn;
sig = [1 0.3; 0.3 1];

%% Data Generation
Sim.Z   = randn(T+burn+1,N);         % instrument
Sim.Y   = zeros(T+burn+1,N);         % initialize data
Sim.X   = zeros(T+burn+1,N);
Sim.epi = zeros(T+burn+1,N);
tmp_phi = nan(1,N);
tmp_bet = nan(1,N);
mu      = rand(1,N);
for i = 1:N
    tmp_phi(i) = phi(G(i));   % assign group coef
    tmp_bet(i) = bet(G(i));
    noise   = mvnrnd([0;0],sig,T+burn+1);
    Sim.X(:,i) = mu(i) + pi*Sim.Z(:,i) + noise(:,1);
    Sim.epi(:,i) = noise(:,2);
end

% generate Y
for t = 2:T+burn+1
    Sim.Y(t,:) = mu + tmp_phi.*Sim.Y(t-1,:)+ tmp_bet.*Sim.X(t,:)+Sim.epi(t,:);
end

% drop first burn+1 obs
Sim.Y = Sim.Y(burn+2:end,:);
Sim.X = Sim.X(burn+2:end,:);
Sim.Z = Sim.Z(burn+2:end,:);
Sim.epi = Sim.epi(burn+2:end,:);


%% Data Preparation
% Take first difference (so that we can instrument Delta yit-1)
FD = Sim.Y - lag(Sim.Y,1);
Sim.reg.LHS = [reshape(FD,N*T,1) reshape(lag(FD,-H),N*T,H)];
Sim.reg.x  = reshape(Sim.X-lag(Sim.X,1),N*T,1);
Sim.reg.c  = reshape(lag(FD,1),N*T,1);

% instrument include y_i,t-2 and z_i,t-1
ylag = reshape(lag(Sim.Y,2),N*T,2);
Sim.reg.zx = reshape(lag(Sim.Z,1),N*T,1);
Sim.reg.zc = ylag(:,2);

idna        = any(isnan([Sim.reg.LHS Sim.reg.x Sim.reg.c Sim.reg.zx Sim.reg.zc]),2);
Sim.reg.LHS = Sim.reg.LHS(~idna,:);
Sim.reg.x   = Sim.reg.x(~idna,:);
Sim.reg.c   = Sim.reg.c(~idna,:);
Sim.reg.zx  = Sim.reg.zx(~idna,:);
Sim.reg.zc  = Sim.reg.zc(~idna,:);

Sim.reg.param.N = N;
Sim.reg.param.T = size(Sim.reg.x,1)/N;
Sim.reg.param.nwtrunc = H+1;
end