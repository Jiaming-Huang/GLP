function par = params(K0,parchoice)
% phi controls persistence
% bet controls size
% parchoice - 1, same size different persistence
%           - 2, different size same persistence

if K0 == 2
    if parchoice == 1
        bet       = [3 3];
        phi        = [0.2 0.6];
    else
        bet         = [1 2];
        phi         = [0.5 0.5];
    end
elseif K0 == 3
    if parchoice == 1
        bet        = [3 3 3];
        phi        = [0.2, 0.6, 0.9];        
    else
        bet        = [1 2 3];
        phi        = [0.5, 0.5, 0.5];
    end
else
    if parchoice == 1
        bet        = [3 3 3 3];
        phi        = [0.2, 0.4, 0.7, 0.9];        
    else
        bet        = [1 2 3 4];
        phi        = [0.5, 0.5, 0.5 0.5];
    end
end

par = [phi;bet];
end