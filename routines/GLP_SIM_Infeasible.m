function [GIRF, GSE, Ubands, Lbands] = GLP_SIM_Infeasible(reg, Gr, FE)
% Infeasible GLP (IGLP) algorithm used for simulation where we know the true group partition
% Specifically, we run pooled panel LP-IV (xtivreg2, fe cluster(id)) given the true group partition

% Related functions:
% GLP_SIM_KnownG0.m: this function takes G0 as given and initializes with true IRs and just run one iteration
% GLP_SIM_UnknownG0.m: this function loops over Ghat=1,...Gmax and select Ghat by IC; since it runs over Ghat neq G0, it uses IND_LP as initial guess
% GLP.m: this is a fully fledged version of GLP that can be used for empirical applications

% INPUT:
%   reg: data (reg.LHS, reg.x, reg.c, reg.zx, reg.zc, reg.param)
%           reg.LHS NT by H dependent variables
%           reg.x, NT by K policy variables whose coefs are to be grouped
%           reg.zx, NT by Lx IV for reg.x (optional, use reg.x if not specified)
%           reg.c, NT by P controls whose coefs vary across i (can be empty)
%           reg.zc NT by Lc IV for reg.c (optional, use reg.c if not specified)
%           reg.param.N, reg.param.T
%           for simplicity, the program is written for balanced panel, but can be easily adapted to unbalanced ones
%   Gr: (true) group partition
%   FE: 1 - fixed effects (within estimator, demean)

% OUTPUT:
%   GIRF: Group IRF, K by 1 by G0 by H matrix
%   GSE: Group IRF, K by 1 by G0 by H matrix
%   Ubands: Upper bound, K by 1 by G0 by H matrix
%   Lbands: Lower bound, K by 1 by G0 by H matrix

T = reg.param.T;
H = size(reg.LHS,2);
K = size(reg.x,2);
G0 = length(unique(Gr));

%% Do Standard panel LP-IV with Group info
GrLong = kron(Gr,ones(T,1));
GIRF = nan(K,1,G0,H);
GSE  = nan(K,1,G0,H);
Ubands  = nan(K,1,G0,H);
Lbands  = nan(K,1,G0,H);

tmp = [];
if isempty(reg.c)
    for g = 1:G0
        tmp.x           = reg.x(GrLong==g,:);
        tmp.c           = [];
        tmp.zx          = reg.zx(GrLong==g,:);
        tmp.zc          = [];
        tmp.LHS         = reg.LHS(GrLong==g,:);
        tmp.param.N     = sum(Gr==g);
        tmp.param.T     = reg.param.T;
        panOut          = panel_LP(tmp,FE);
        GIRF(:,:,g,:)   = panOut.IR;
        GSE(:,:,g,:)    = panOut.IRse;
        Ubands(:,:,g,:) = panOut.IRUb;
        Lbands(:,:,g,:) = panOut.IRLb;
    end
else
    for g = 1:G0
        tmp.x           = reg.x(GrLong==g,:);
        tmp.c           = reg.c(GrLong==g,:);
        tmp.zx          = reg.zx(GrLong==g,:);
        tmp.zc          = reg.zc(GrLong==g,:);
        tmp.LHS         = reg.LHS(GrLong==g,:);
        tmp.param.N     = sum(Gr==g);
        tmp.param.T     = reg.param.T;
        panOut          = panel_LP(tmp,FE);
        GIRF(:,:,g,:)   = panOut.IR;
        GSE(:,:,g,:)    = panOut.IRse;
        Ubands(:,:,g,:) = panOut.IRUb;
        Lbands(:,:,g,:) = panOut.IRLb;
    end
end

end