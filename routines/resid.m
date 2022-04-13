function [dif, converge] = resid(b_old,b_new,Q_old,Q_new,tol)
% Compute whether the iteration converges
% Inputs:
% b_old, b_new: K by 1 by G by H matrix of group coefs
% Q_old, Q_new: obj function

% Outputs:
% dif is the max of the two criteria
% converge is a dummy indicator 1(converged) 0 (not)
G0 = size(b_old,3);
a_nom = 0;
a_den = 0;
for i = 1:G0
    old = b_old(:,:,i,:);
    new = b_new(:,:,i,:);
    a_nom = a_nom + norm(old(:)-new(:));
    a_den = a_den + norm(old(:));
end

converge = 0;
dif = max(abs(Q_new-Q_old), a_nom/(a_den+0.001));

if dif < tol
    converge = 1;
end
end