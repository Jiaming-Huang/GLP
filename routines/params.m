function par = params(G0,parchoice)
% phi controls persistence
% bet controls size
% parchoice - 1, same size different persistence
%           - 2, different size same persistence
if G0 == 1
    bet = 1;
    phi = 0.5;
elseif G0 == 2
    if parchoice == 1
        bet       = [3 3];
        phi        = [0.2 0.6];
    else
        bet         = [1 2];
        phi         = [0.5 0.5];
    end
elseif G0 == 3
    if parchoice == 1
        bet        = [3 3 3];
        phi        = [0.2, 0.6, 0.9];        
    elseif parchoice == 2
        bet        = [1 2 3];
        phi        = [0.5, 0.5, 0.5];
    end
end

par = [phi;bet];
end