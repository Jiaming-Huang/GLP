function yt   = prepare_missing(rawdata,tcode)
% =========================================================================
% DESCRIPTION: 
% This function transforms raw data based on each series' transformation
% code.
%
% -------------------------------------------------------------------------
% INPUT:
%           rawdata     = raw data 
%           tcode       = transformation codes for each series
%
% OUTPUT: 
%           yt          = transformed data
%
% -------------------------------------------------------------------------
% SUBFUNCTION:
%           transxf:    transforms a single series as specified by a 
%                       given transfromation code
%
% =========================================================================
% APPLY TRANSFORMATION:
% Initialize output variable
yt        = [];                                     

% Number of series kept
N = size(rawdata,2);                         

% Perform transformation using subfunction transxf (see below for details)
for i = 1:N
    dum = transx(rawdata(:,i),tcode(i));
    yt    = [yt, dum];
end

end
