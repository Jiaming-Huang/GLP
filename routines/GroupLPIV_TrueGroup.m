function [GIRF, GSE] = GroupLPIV_TrueGroup(reg, Group,FE)
T = reg.param.T;
H = size(reg.LHS,2);
K = length(unique(Group));
%% Do Standard panel LP-IV with Group info
gtmp = kron(Group,ones(T,1));
GIRF = nan(K,H);
GSE = nan(K,H);
tmp = [];
for k = 1:K
    tmp.y = reg.y(gtmp==k,:);
    tmp.x = reg.x(gtmp==k,:);
    tmp.z = reg.z(gtmp==k,:);
    tmp.LHS = reg.LHS(gtmp==k,:);%cumsum(reg.LHS(gtmp==k,:),2);
    tmp.control = reg.control(gtmp==k,:);
    tmp.param.N = sum(Group==k);
    tmp.param.T = reg.param.T;
    [GIRF(k,:), ~, GSE(k,:)] = panel_LP1(tmp,FE);
end

end

function x_de = gdemean(x,param)
x_de = nan(size(x,1),size(x,2));
for i = 1: param.N
    % for each unit
    xi = x(param.T*(i-1)+1:param.T*i,:);
    x_mean = mean(xi);
    x_de(param.T*(i-1)+1:param.T*i,:) = xi - repmat(x_mean,param.T,1);
end
end