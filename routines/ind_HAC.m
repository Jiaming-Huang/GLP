function v = ind_HAC(g,kappa)
%{

Taken from SW, EJ 2018. Compute HAC variance
Input:
     g = X*e
     kappa =truncation parameter (nma=0, White SEs)

Output:
    v = Robust estimate of covariance matrix of beta (kxk)

%}
v=zeros(size(g,2),size(g,2));


% Form Kernel
kern=zeros(kappa+1,1);

for ii = 0:kappa
    kern(ii+1,1)=1;
    if kappa > 0
        kern(ii+1,1)=(1-(ii/(kappa+1)));
    end
end

% Form Hetero-Serial Correlation Robust Covariance Matrix
for ii = -kappa:kappa
    if ii <= 0
        r1=1;
        r2=size(g,1)+ii;
    else
        r1=1+ii;
        r2=size(g,1);
    end
    v=v + kern(abs(ii)+1,1)*(g(r1:r2,:)'*g(r1-ii:r2-ii,:));
end


end