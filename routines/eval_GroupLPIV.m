function [Accuracy, RMSE, Permutation, Ng] = eval_GroupLPIV(Group, IR_TRUE, GIRF)
% Input:
% Group = [GT P]: GT is the ground truth, P is the predicted classification
%                 both of them are N by 1 vectors
% IRF_TRUE: N by H matrix of true IRs
% GIRF: estimated group IRs

%% Prepare
% ground truth partition
GT = Group(:,1);
% predicted group partition
P  = Group(:,2);

% number of entities
N  = length(GT);
H  = size(IR_TRUE,2);
% true group number
G0 = length(unique(GT));
% supplied group number
GP  = length(unique(P));

% estimated IRs, create it into N by H
GIR_EST = nan(size(IR_TRUE));
for g = 1:GP
    for h = 1:H
    GIR_EST(P==g,h) = GIRF(g,h);
    end
end


%% Find permutation that maximize classification accuracy

% get all possible permutations
v  = perms([1:GP]);
perm = cell(size(v,1),1);
ac_perm = nan(size(v,1),1);

for i = 1: size(v,1)
    % for each possible permutation
    % relabel our Pnew
    tmp = zeros(N,1);
    for g = 1:GP
        tmp = tmp +(P == v(i,g) )*g;
    end
    perm{i,1} = tmp; % store it for later output
    
    % now we got the new label, compute accuracy
    ac_perm(i) = sum(GT==tmp)/N;
    
end

% get the permutation
[Accuracy,idx] = max(ac_perm);

Label = perm{idx,1};

Permutation = v(idx,:);

% now reassign GIRF
Ng = nan(1,GP);
for g = 1:GP
    Ng(g) = sum(Label==g);   % used to weight the RMSE
end

%% Compute RMSE
RMSE = sqrt(mean(mean((GIR_EST - IR_TRUE).^2)));

end