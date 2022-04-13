function Sim = DGP_NoGroup(N,T,DGPsetup)
% generate simulation data
% --------------------------- MODEL --------------------------------
% y_it = mu_i + phi_g*y_it-1 + beta_g*x_it + ep_it
% x_it = mu_i + pi*z_it + u_it
% mu_i ~ U(0,1)
% z_it, xi_it are iid N(0,1)
% u_it, ep_it are bivarate normal, with Sig=[1 0.3; 0.3 1]

% --------------------------- INPUT --------------------------------
% N,T  - size of the panel (balanced)
% DGPsetup: a data struct of parameters, containing
%           1) K    - size of X
%           2) H    - impulse responses horizons
%           3) burn - number of burning periods

% --------------------------- OUTPUT --------------------------------
% Sim: a data struct ready for estimation, containing
%           1) Z, Y, X - raw data series for instruments Z, 
%                               data Y and X
%           2) reg        - reshaped data for estimation
%           3) param      - N and T

%% Unpack Parameters
bet  = 1 + (3-1)*rand(1,N);       % U(1, 3)
phi  = 0.1 + (0.9-0.1)*rand(1,N); % U(0.1, 0.9)
pi  = 0.7;
K    = DGPsetup.K;
H    = DGPsetup.H;
burn = DGPsetup.burn;
sig  = [1 0.3; 0.3 1];

IR_TRUE = nan(K,1,N,H+1);
for i =1:N
    IR_TRUE(:,:,i,:) = bet(i)*(phi(i).^[0:H]);
end
         
%% Data Generation
Sim.Z   = randn(T+burn+1,N);         % instrument
Sim.Y   = zeros(T+burn+1,N);         % initialize data
Sim.X   = zeros(T+burn+1,N);
Sim.epi = zeros(T+burn+1,N);
mu      = rand(1,N);

for i = 1:N
    noise        = mvnrnd([0;0],sig,T+burn+1);
    Sim.X(:,i)   = mu(i) + pi*Sim.Z(:,i) + noise(:,1);
    Sim.epi(:,i) = noise(:,2);
end

% generate Y
for t = 2:T+burn+1
    Sim.Y(t,:) = mu + phi.*Sim.Y(t-1,:)+ bet.*Sim.X(t,:)+Sim.epi(t,:);
end

% drop first burn+1 obs
Sim.Y   = Sim.Y(burn+2:end,:);
Sim.X   = Sim.X(burn+2:end,:);
Sim.Z   = Sim.Z(burn+2:end,:);
Sim.epi = Sim.epi(burn+2:end,:);


%% Data Preparation
Sim.reg.LHS = [reshape(Sim.Y,N*T,1) reshape(lag(Sim.Y,-H),N*T,H)];
Sim.reg.x   = reshape(Sim.X,N*T,1);
Sim.reg.zx  = reshape(Sim.Z,N*T,1);
Sim.reg.c   = reshape(lag(Sim.Y,1),N*T,1);
Sim.reg.zc  = Sim.reg.c;

idna        = any(isnan([Sim.reg.LHS Sim.reg.x Sim.reg.c Sim.reg.zx Sim.reg.zc]),2);
Sim.reg.LHS = Sim.reg.LHS(~idna,:);
Sim.reg.x   = Sim.reg.x(~idna,:);
Sim.reg.c   = Sim.reg.c(~idna,:);
Sim.reg.zx  = Sim.reg.zx(~idna,:);
Sim.reg.zc  = Sim.reg.zc(~idna,:);

Sim.reg.param.N = N;
Sim.reg.param.T = size(Sim.reg.x,1)/N;
Sim.reg.param.nwtrunc = H+1;

Sim.phi = phi;
Sim.bet = bet;
Sim.IR_TRUE= IR_TRUE;
end