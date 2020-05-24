function RA = adjusted_RSquare_weight(X,Y,W)
    WX = X .* W.^(1/2);
    WY = Y .* W.^(1/2);
    P = least_squares_estimation(WX,WY);
    [r c] = size(X);
    Y_ = MEAN(Y);

    SSR = sum(W.*(X*P - Y_).^2);
    SST = sum(W.*(Y - Y_).^2);
    SSE = sum(W.*(Y - X*P).^2);
    sqrt(SSE/29)
    %e_ = sum((WY - WX*P).^2)
   % t_ = sum((WY - Y_).^2)
   % r_ = sum((WX*P - Y_).^2)
   % (e_ + r_) - t_
    
    R = SSR/SST;
  %  R1 = 1 - SSR/SST
  % E = SSE / SST;
   % (SSR + SSE) - SST
    RA = 1 - (1-R)*( (r - 1)/(r - c) );
end