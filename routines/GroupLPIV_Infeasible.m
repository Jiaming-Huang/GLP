function [GIRF, GSE] = GroupLPIV_Infeasible(reg, Gr, FE)
% infeasible GLP
% that is, pooled panel LP-IV given the true group partition

T = reg.param.T;
H = size(reg.LHS,2);
G0 = length(unique(Gr));

%% Do Standard panel LP-IV with Group info
GrLong = kron(Gr,ones(T,1));
GIRF = nan(G0,H);
GSE = nan(G0,H);
tmp = [];
if isempty(reg.control)
    for g = 1:G0
        tmp.y = reg.y(GrLong==g,:);
        tmp.x = reg.x(GrLong==g,:);
        tmp.z = reg.z(GrLong==g,:);
        tmp.LHS = reg.LHS(GrLong==g,:);
        tmp.control = [];
        tmp.param.N = sum(Gr==g);
        tmp.param.T = reg.param.T;
        [GIRF(g,:), GSE(g,:)] = panel_LP(tmp,FE);
    end
else
    for g = 1:G0
        tmp.y = reg.y(GrLong==g,:);
        tmp.x = reg.x(GrLong==g,:);
        tmp.z = reg.z(GrLong==g,:);
        tmp.LHS = reg.LHS(GrLong==g,:);
        tmp.control = reg.control(GrLong==g,:);
        tmp.param.N = sum(Gr==g);
        tmp.param.T = reg.param.T;
        [GIRF(g,:), GSE(g,:)] = panel_LP(tmp,FE);
    end
end
end