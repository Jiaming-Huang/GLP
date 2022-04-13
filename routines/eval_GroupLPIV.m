function [Accuracy, MSE, BR, Permutation, Ng, Gr_relabel, GIRF_relabel, GSE_relabel] = eval_GroupLPIV(Group, IR_TRUE, GIRF, GSE, indSE)
% Input:
% Group = [GT P]: GT is the ground truth, P is the predicted classification
%                 both of them are N by 1 vectors
% IR_TRUE: K by H by N matrix of true IRs
% GIRF: K by H by G_hat estimated group IRs

%% Prepare
% ground truth partition
Gr_True = Group(:,1);
% predicted group partition
Gr_hat  = Group(:,2);

% number of entities
N  = length(Gr_True);

% supplied group number
G_hat  = length(unique(Gr_hat));

% estimated IRs, create it into K by 1 by N by H
Ng      = sum(Gr_hat==[1:G_hat]);
GIR_EST = nan(size(IR_TRUE));
GSE_EST = nan(size(IR_TRUE));
for g = 1:G_hat
    GIR_EST(:,:,Gr_hat==g,:) = repmat(GIRF(:,:,g,:),1,1,Ng(g),1);
    GSE_EST(:,:,Gr_hat==g,:) = repmat(GSE(:,:,g,:),1,1,Ng(g),1);
end

%% Find permutation that maximize classification accuracy

% get all possible permutations
v  = perms([1:G_hat]);
perm = cell(size(v,1),1);
ac_perm = nan(size(v,1),1);

for i = 1: size(v,1)
    % for each possible permutation
    % relabel our Pnew
    tmp = zeros(N,1);
    for g = 1:G_hat
        tmp = tmp +(Gr_hat == v(i,g) )*g;
    end
    perm{i,1} = tmp; % store it for later output
    
    % now we got the new label, compute accuracy
    ac_perm(i) = sum(Gr_True==tmp)/N;
    
end

% get the permutation
[Accuracy,idx] = max(ac_perm);

Permutation = v(idx,:);

% now reassign the estimated objects
Ng = Ng(Permutation);
Gr_relabel   = perm{idx,1};
GIRF_relabel = GIRF(:,:,Permutation,:);
GSE_relabel  = GSE(:,:,Permutation,:);

%% Compute MSE
err2 = (GIR_EST - IR_TRUE).^2;
MSE = mean(err2(:));

%% Compare Band Ratio
seRatio = GSE_EST./indSE;
BR = mean(seRatio(:));
end