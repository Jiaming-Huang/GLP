function Sim = DGP_CompareZ(N,T,DGPsetup)
% generate simulation data
% --------------------------- MODEL --------------------------------
% y_it = mu_i  + ep_it + phi_g*ep_it-1
% estimate y_it+1 = beta* y_it + u_it
% case 1: Arellano-Bond type
% do first difference and instrument delta y_it with y_it-1
% case 2: we have an external IV
% z_it = ep_it + lam*xi_it
% mu_i ~ U(0,1)
% ep_it, xi_it are iid N(0,1)

% --------------------------- INPUT --------------------------------
% N,T  - size of the panel (balanced)
% DGPsetup: a data struct of parameters, containing
%           1) phi  - persistence of impulse responses, group specific
%           2) G    - N by 1 vector indicating group number
%           3) H    - impulse responses horizons
%           4) burn - number of burning periods

% --------------------------- OUTPUT --------------------------------
% Sim: a data struct ready for estimation, containing
%           1) S, Z, Y, X - raw data series for shocks S, instruments Z, 
%                               data Y and X
%           2) regAH      - reshaped data for estimation (Arellano-Bond)
%              reg        - reshaped data for estimation (external IV)
%           3) param      - N and T

%% Unpack Parameters
phi = DGPsetup.par(1,:);
lambda = DGPsetup.lambda;
G   = DGPsetup.G;
burn= DGPsetup.burn;

%% Data Generation
Sim.epi = randn(T+burn+1,N);           % Shocks
Sim.Z   = Sim.epi + lambda*randn(T+burn+1,N); % instrument
Sim.Y   = zeros(T+burn+1,N);           % initialize data

tmp_phi = nan(1,N);
mu      = rand(1,N);
for i = 1:N
    tmp_phi(i) = phi(G(i));   % assign group coef
end

% generate Y
for t = 2:T+burn+1
    Sim.Y(t,:) = mu + Sim.epi(t,:) + tmp_phi.*Sim.epi(t-1,:);
end

% drop first burn+1 obs
Sim.Y = Sim.Y(burn+2:end,:);
Sim.Z = Sim.Z(burn+2:end,:);
Sim.epi = Sim.epi(burn+2:end,:);


%% Data Preparation
% external IV 
Sim.reg.y = reshape(Sim.Y,N*T,1);
Sim.reg.LHS = [reshape(lag(Sim.Y,-1),N*T,1)]; % y_i,t+1

Sim.reg.x = reshape(Sim.Y,N*T,1);
Sim.reg.z = reshape(Sim.Z,N*T,1);
Sim.reg.control = [];

idna = logical(sum(isnan([Sim.reg.x Sim.reg.LHS]),2));
Sim.reg.LHS = Sim.reg.LHS(~idna,:);
Sim.reg.x = Sim.reg.x(~idna,:);
Sim.reg.y = Sim.reg.y(~idna,:);
Sim.reg.z = Sim.reg.z(~idna,:);

Sim.reg.param.N = N;
Sim.reg.param.T = size(Sim.reg.x,1)/N;

% Anderson-Hsiao (1982)
FD = Sim.Y - lag(Sim.Y,1);
Sim.regAH.LHS = reshape(lag(FD,-1),N*T,1);
Sim.regAH.x = reshape(FD,N*T,1);
Sim.regAH.z = reshape(lagmatrix(Sim.Y,1),N*T,1);
Sim.regAH.control = [];

idna = logical(sum(isnan([Sim.regAH.LHS Sim.regAH.x Sim.regAH.z]),2));
Sim.regAH.LHS = Sim.regAH.LHS(~idna,:);
Sim.regAH.x = Sim.regAH.x(~idna,:);
Sim.regAH.z = Sim.regAH.z(~idna,:);

Sim.regAH.param.N = N;
Sim.regAH.param.T = size(Sim.regAH.x,1)/N;

end