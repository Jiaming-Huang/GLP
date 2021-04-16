function [dif, converge] = resid(b_old,b_new,Q_old,Q_new,tol)
% Compute whether the iteration converges
% Inputs:
% b_old, b_new: K by h matrix of group coefs
% Q_old, Q_new: obj function

% Outputs:
% dif is the max of the two criteria
% converge is a dummy indicator 1(converged) 0 (not)

a_nom = sum(sum( vecnorm( abs( b_old - b_new ), 2, 2 ) ));
a_den = sum(sum( vecnorm( abs( b_old ), 2, 2 ) )) + 0.001;

converge = 0;
dif = max(abs(Q_new-Q_old), a_nom/a_den);

if dif < tol
    converge = 1;
end
end