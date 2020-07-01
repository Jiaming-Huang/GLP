function res = constructBootSamplesWild(VAR,modelSpec)
%returns wild bootstrap set of endogenous in main VAR (Y); endogenous in
%VAR with fed funds used for long-run expectations (X); instrument (Z)

% wild bootstrap:
% randomly switch sign of the reduced form innovations (same for z)

% unpack variables
u           = VAR.resid;
b           = VAR.b;
nL          = VAR.nLags;
ismpl       = VAR.ismpl;
iScheme     = modelSpec.identification;

[nT,K]      = size(u);



% bootstrapped residuals
bSwitch     = 2*(rand(nT,1)>.5) - 1;
bootU       = u.*(bSwitch*ones(1,K));



% bootstrapped samples
% initialize Y
t0          = find(ismpl,1);
t1          = find(ismpl,1,'last');
bootY       = nan(nT,K);
bootY(t0-nL:t0-1,:)=VAR.Ylag(1:nL,end-K+1:end);  % keeping the first nL obs
Ylag         =VAR.Ylag(1,:);

if modelSpec.iconst ==1
    for t=t0:t1
        % construct new obs from nL+1 to end        
        bootY(t,:) =[1 Ylag]*b + bootU(t,:);
        Ylag = [bootY(t,:) Ylag(1:end-K)];
    end
else
    for t=t0:t1
        % construct new obs from nL+1 to end
        bootY(t,:) =Ylag*b + bootU(t,:);
        Ylag = [bootY(t,:) Ylag(1:end-K)];
    end
end

res.Y       = bootY;

if contains(iScheme,'IVSVAR')
    z           = modelSpec.ins;
    res.Z       = z.*bSwitch;
end

end
