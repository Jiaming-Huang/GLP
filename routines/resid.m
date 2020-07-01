function [dif, d] = resid(a_old,a_new,Q_old,Q_new,tol)
% Compute whether the iteration converges
% Inputs:
% a_old, a_new: K by h matrix of group coefs
% Q_old, Q_new: obj function

% Outputs:
% dif is the max of the two criteria
% d is a dummy indicator 1(converged) 0 (not)

a_nom = sum(sum( vecnorm( abs( a_old - a_new ), 2, 2 ) ));
a_den = sum(sum( vecnorm( abs( a_old ), 2, 2 ) )) + 0.001;

d = 0;
dif = max(abs(Q_new-Q_old), a_nom/a_den);

if dif < tol
    d = 1;
end
end