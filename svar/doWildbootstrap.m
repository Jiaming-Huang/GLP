function VARbs = doWildbootstrap(VAR,nboot,clevel)

% Wild Bootstrap
res = detrend(VAR.res,'constant');
IRS = zeros(size(VAR.irs(:),1),nboot,size(VAR.irs,3));

for jj = 1:nboot
    rr = 1-2*(rand(VAR.T,1)>0.5);
    resb = (res.*(rr*ones(1,VAR.n)))';
    
    varsb = zeros(VAR.p+VAR.T,VAR.n);
    varsb(1:VAR.p,:)=VAR.vars(1:VAR.p,:);
    
    for j=VAR.p+1:VAR.p+VAR.T
        
        lvars = (varsb(j-1:-1:j-VAR.p,:))';
        varsb(j,:) = lvars(:)'*VAR.bet(1:VAR.p*VAR.n,:)+VAR.DET(j,:)*VAR.bet(VAR.p*VAR.n+1:end,:)+resb(:,j-VAR.p)';
    end
    
    VARBS = VAR;
    VARBS.vars = varsb;
    
    VARBS.proxy = [VAR.proxy(1:VAR.p,:); VAR.m.*(rr*ones(1,VAR.k))];
    VARBS = doProxySVAR(VARBS);
    
    
    for i=1:size(VARBS.irs,3)
        irs = VARBS.irs(:,:,i);
        IRS(:,jj,i) = irs(:);
    end
    
end

% Confidence Bands
%%%%%%%%%%%%%%%%%%

for jj = 1:length(clevel)
    for i=1:size(IRS,3)
        VARbs.irsH(:,:,jj,i)=reshape(quantile(IRS(:,:,i)',(1-clevel(jj)/100)/2),VAR.irhor, size(VAR.irs,2));
        VARbs.irsL(:,:,jj,i)=reshape(quantile(IRS(:,:,i)',1-(1-clevel(jj)/100)/2),VAR.irhor, size(VAR.irs,2));
    end
end


