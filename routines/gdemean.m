function x_de = gdemean(x,N,T)
% designed for balanced panel, x is NT by K matrix
[NT,K] = size(x);
xt = reshape(x',K,T,N);
x_de = reshape(xt - mean(xt,2),K,NT)';
end