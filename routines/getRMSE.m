function rmse = getRMSE(Gr_Est, IR_TRUE, GIRF)
    
% predicted group partition
P  = Gr_Est;

H  = size(IR_TRUE,2);
% supplied group number
GP  = length(unique(P));

% estimated IRs, create it into N by H
GIR_EST = nan(size(IR_TRUE));
for g = 1:GP
    for h = 1:H
    GIR_EST(P==g,h) = GIRF(g,h);
    end
end

%% Compute RMSE
rmse = sqrt(mean(mean((GIR_EST - IR_TRUE).^2)));
end