function [b, se, F] = ind_LP(reg)

%% unpack variables
LHS = reg.LHS;
control = reg.control;
x   = reg.x;
z   = reg.z;
T   = reg.param.T;
N   = reg.param.N;
H   = size(LHS,2);

b = nan(N,H);
se = nan(N,H);
F = nan(N,1);

if isempty(z)
    %fprintf('Model: Local Projection for each entity (without IV) \n')
    
    for i = 1:N
        % construct regression variables
        Y = LHS(T*(i-1)+1:T*i,:);  %y(T*(i-1)+1:T*i,:)
        X = [x(T*(i-1)+1:T*i,:)  ones(T,1) control(T*(i-1)+1:T*i,:)];
        
        bhat = (X'*X)\(X'*Y);
        ehat = Y - X*bhat;
        b(i,:) = bhat(1,:);
        
        % compute HAC SE
        for h = 1:H
            g = X.*repmat(ehat(:,h),1,size(X,2));
            v_hac = ind_HAC(g,H+1);
            vbeta=(X'*X)\v_hac/(X'*X);
            tmp = sqrt(diag(vbeta));
            se(i,h) = tmp(1);
        end
        
    end
    F = [];
    
    
else
    %fprintf('Model: Local Projection for each entity (with IV) \n')
    
    for i = 1:N
        Y    = LHS(T*(i-1)+1:T*i,:);  % y(T*(i-1)+1:T*i,:)
        X    = [x(T*(i-1)+1:T*i,:) ones(T,1) control(T*(i-1)+1:T*i,:)];
        Z    = [z(T*(i-1)+1:T*i,:) ones(T,1) control(T*(i-1)+1:T*i,:)];
        
        % 2SLS
        Xhat = Z*((Z'*Z)\(Z'*X)); % first stage
        bhat = (Xhat'*Xhat)\(Xhat'*Y);
        ehat = Y - X*bhat;
        b(i,:) = bhat(1,:);
        
        % compute HAC SE
        for h = 1:size(Y,2)
            g = Xhat.*repmat(ehat(:,h),1,size(X,2));
            v_hac = ind_HAC(g,H+1);
            vbeta=(Xhat'*Xhat)\v_hac/(Xhat'*Xhat);
            tmp = sqrt(diag(vbeta));
            se(i,h) = tmp(1);
        end
        
        % Compute F-statistics
        X = X(:,1);
        gam = (Z'*Z)\(Z'*X);
        uhat = X-Z*gam;
        g = Z.*repmat(uhat,1,size(Z,2));
        v_hac = ind_HAC(g,h+1);
        vbeta=(Z'*Z)\v_hac/(Z'*Z);
        tmp = sqrt(diag(vbeta));
        F(i) = (gam(1)/tmp(1))^2;
    end
end

end
