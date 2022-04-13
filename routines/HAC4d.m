function v = HAC4d(g,kappa)
%{

Adapted from SW, EJ 2018. Compute HAC variance
Input:
     g = X*e
     kappa =truncation parameter (nma=0, White SEs)

Output:
    v = Robust estimate of covariance matrix of beta (kxk)

%}

[T,K,N,H] = size(g);
v = zeros(K,K,N,H);
% Form Kernel
kern = 1-[1:kappa]/(kappa+1);

% Form Hetero-Serial Correlation Robust Covariance Matrix
gt = pagetranspose(g);
for i = 1:kappa
    gg = pagemtimes(gt(:,1+i:T,:,:),g(1:T-i,:,:,:));
    v = v + kern(1,i)*(gg+pagetranspose(gg));
end
v = v+pagemtimes(gt,g);
end