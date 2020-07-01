function VARci = doProxySVARci(VAR)

% method  1: Mertens and Ravn (2014) wild bootstrap 
%         2: Delta Method 
%         3: Jentsch Lunsford MBB 

% Dimension of VARci.irsH and VARci.irsL: 
% 1: horizon 2: # of variables 3: clevel 4: shock (if more than one shock with instrument)

cLevel = VAR.cLevel;
bootMethod = VAR.bootMethod;
nBoot  = VAR.nBoot;

VARci.irsL = NaN*zeros([size(VAR.irs) length(cLevel)]);
VARci.irsH = NaN*zeros([size(VAR.irs) length(cLevel)]);

if bootMethod == 1    
    VARci = doWildbootstrap(VAR,nBoot,cLevel); 
elseif bootMethod == 2
    VARci = doDeltaMethod(VAR,cLevel,VAR.NWlags); % Not available for k>1
elseif bootMethod == 3
    BlockSize = 5;
    VARci = doMBBbootstrap(VAR,nBoot,cLevel,BlockSize);    
end

if VAR.k == 1
[~,What,~] = CovAhat_Sigmahat_Gamma(VAR.p,VAR.X(:,[end end-size(VAR.DET,2)+1:end-1 1:VAR.n*VAR.p]),VAR.m,VAR.res',VAR.NWlags);                
 Gamma       = VAR.res'*VAR.m./VAR.T; 
VARci.Waldstat = (((VAR.T^.5)*Gamma(1,1))^2)/What(((VAR.n^2)*VAR.p)+1,((VAR.n^2)*VAR.p)+1);
end
