%复决定系数
function R = R_square(Z,Y)
    [r c] = size(Z);
    P = least_squares_estimation(Z,Y);
    Y_ = MEAN(Y);
    
    SSR = sum((Z*P - Y_).^2);
    SST = sum((Y - Y_).^2);
    R = SSR/SST;
end