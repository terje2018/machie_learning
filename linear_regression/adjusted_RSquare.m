%调整决定系数，判定拟合程度
function RA = adjusted_RSquare(Z,Y)
    [r c] = size(Z);
    P = least_squares_estimation(Z,Y);
    Y_ = MEAN(Y);
    
    SSR = sum((Z*P - Y_).^2);
    SST = sum((Y - Y_).^2);
    SSE = sum((Y - Z*P).^2);
    R = SSR/SST;
    E = SSE / SST;

    (SSR + SSE) - SST;
    
    RA = 1 - (1-R)*( (r - 1)/(r - c) );
end