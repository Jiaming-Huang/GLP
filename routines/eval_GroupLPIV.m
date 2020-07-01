function [Accuracy, RMSE, BIAS, Permutation, Ng] = eval_GroupLPIV(Group, IRF, GIRF)

% Input: Group = [GT P] GT is the ground truth, P is the predicted classification
% both of them are N by 1 vectors
% IRF and IRF_est are both K0 by H

%% Prepare
GT = Group(:,1);
P  = Group(:,2);
N  = length(GT);
K0 = length(unique(GT));
K  = length(unique(P));
v  = perms([1:K]);

perm = cell(size(v,1),1);
ac_perm = nan(size(v,1),1);

%% Find permutation that maximize accuracy
for i = 1: size(v,1)
    % for each possible permutation
    % relabel our Pnew
    tmp = zeros(N,1);
    for k = 1:K
        tmp = tmp +(P == v(i,k) )*k;
    end
    perm{i,1} = tmp; % store it for later output
    
    % now we got the new label, compute accuracy
    ac_perm(i) = sum(GT==tmp)/N;
    
end

[Accuracy,idx] = max(ac_perm);

Label = perm{idx,1};

Permutation = v(idx,:);

%% Based on the new label, compute RMSE
GIRF = GIRF(Permutation,:);
Ng = nan(1,K);
for k = 1:K
    Ng(k) = sum(Label==k);   % used to weight the RMSE
end


SE = nan(K,1);
if K0 >= K   % we guess the true group number
    SE =  mean((IRF(1:K,:) - GIRF).^2,2);
    idx = [];
else
    SE(1:K0) = mean( (IRF - GIRF(1:K0,:) ).^2 , 2 );
    idx = nan(K-K0,1);
    for k = K0+1:K
        % for each extra IRF, find the true IRF that minimizes sum of
        % squared errors
        tmp = nan(K0,1);
        for kk = 1:K0
            tmp(kk) = mean( (IRF(kk,:) - GIRF(k,:) ).^2 );
        end
        [SE(k),idx(K-k+1,1)] = min(tmp);
    end
end


if K>K0
    BIAS = (Ng./N)'.*(GIRF - [IRF;IRF(idx,:)]);
    %MSE = [(Ng(1:K0)./N) *SE(1:K0) (Ng(K0+1:K)./N) *SE(K0+1:K)];
else
    BIAS = (Ng./N)'.*(GIRF - IRF(1:K,:));
    %MSE = Ng./N*SE;
end
MSE = Ng./N*SE;
RMSE = sqrt( MSE );
end