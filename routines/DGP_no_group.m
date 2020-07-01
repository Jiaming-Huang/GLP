function Sim = DGP_no_group(N,T,DGPsetup)
% generate simulation data
% --------------------------- MODEL --------------------------------
% y_it = mu_i + phi_g*y_it-1 + beta_g*x_it + ep_it
% x_it = mu_i + s_it + u_it
% z_it = s_it + xi_it
% mu_i ~ U(0,1)
% s_it, xi_it are iid N(0,1)
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
H   = DGPsetup.H;
burn= DGPsetup.burn;
sig = [1 0.3; 0.3 1];

%% Data Generation
Sim.S   = randn(T+burn+1,N);         % Shocks
Sim.Z   = Sim.S + randn(T+burn+1,N); % instrument
Sim.Y   = zeros(T+burn+1,N);         % initialize data
Sim.X   = zeros(T+burn+1,N);
Sim.epi = zeros(T+burn+1,N);
mu      = rand(1,N);

for i = 1:N
    noise   = mvnrnd([0;0],sig,T+burn+1);
    Sim.X(:,i) = mu(i) + Sim.S(:,i) + noise(:,1);
    Sim.epi(:,i) = noise(:,2);
end

% generate Y
for t = 2:T+burn+1
    Sim.Y(t,:) = mu + phi.*Sim.Y(t-1,:)+ bet.*Sim.X(t,:)+Sim.epi(t,:);
end

% drop first burn+1 obs
Sim.Y = Sim.Y(burn+2:end,:);
Sim.X = Sim.X(burn+2:end,:);
Sim.Z = Sim.Z(burn+2:end,:);
Sim.epi = Sim.epi(burn+2:end,:);


%% Data Preparation
Sim.reg.y = reshape(Sim.Y,N*T,1);
Sim.reg.LHS = [Sim.reg.y reshape(lag(Sim.Y,-H),N*T,H)]; % including h=0


Sim.reg.x = reshape(Sim.X,N*T,1);
Sim.reg.z = reshape(Sim.Z,N*T,1);
Sim.reg.control = [reshape(lag(Sim.Y,1),N*T,1)];



idna = logical(sum(isnan([Sim.reg.control Sim.reg.LHS]),2));
Sim.reg.LHS = Sim.reg.LHS(~idna,:);
Sim.reg.x = Sim.reg.x(~idna,:);
Sim.reg.y = Sim.reg.y(~idna,:);
Sim.reg.z = Sim.reg.z(~idna,:);
Sim.reg.control = Sim.reg.control(~idna,:);

Sim.reg.param.N = N;
Sim.reg.param.T = size(Sim.reg.x,1)/N;
end