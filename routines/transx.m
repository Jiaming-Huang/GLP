function y=transx(x,tcode)
% =========================================================================
% DESCRIPTION:
% This function transforms a single series (in a column vector)as specified
% by a given transfromation code.
%
% -------------------------------------------------------------------------
% INPUT:
%           x       = series (in a column vector) to be transformed
%           tcode   = transformation code (1-7)
%
% OUTPUT:   
%           y       = transformed series (as a column vector)
%
% =========================================================================
% SETUP:
% Value close to zero 
small=1e-10;

% Allocate output variable
T = size(x,1);
y = nan(T,1);

% =========================================================================
% TRANSFORMATION: 
% Determine case 1-7 by transformation code
switch(tcode)
    
  case 1 % Level (i.e. no transformation): x(t)
    y = x;

  case 2 % First difference: x(t)-x(t-1)
    y(2:T) = x(2:T)-x(1:T-1);
  
  case 3 % Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
    y(3:T) = x(3:T)-2*x(2:T-1) + x(1:T-2);

  case 4 % Natural log: ln(x)
    if min(x) > small
        y = log(x);
    else
        error(message('Transformation code is wrong: taking logs for negatives!'));
    end
  
  case 5 % First difference of natural log: ln(x)-ln(x-1)
    if min(x) > small
        x = log(x);
        y(2:T) = 100*(x(2:T)-x(1:T-1));
    else
        error(message('Transformation code is wrong: taking logs for negatives!'));
    end
  
  case 6 % Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
    if min(x) > small
        x = log(x);
        y(3:T) = x(3:T) - 2*x(2:T-1) + x(1:T-2);
    else
        error(message('Transformation code is wrong: taking logs for negatives!'));
    end
  
  case 7 % Percent change: (x(t)/x(t-1)-1)
    y(2:T)=(x(2:T)-x(1:T-1))./x(1:T-1);

end

end
