%显著性检测，判定是否可用于多元线性回归模型
function R = mutil_linear_significance_test(Z,Y,a)
    [r c] = size(Z);
    P = least_squares_estimation(Z,Y)
    Y_ = MEAN(Y);
    
    SSR = sum((Z * P - Y_).^2) / (c - 1);
    SSE = sum((Y - Z * P).^2) / (r - c);
    F = SSR / SSE
    
    f = finv(1 - a,c - 1,r - c);
    if f >= F
        R = 0;
    else
        R = 1;
    end
end