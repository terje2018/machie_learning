function R = mutil_linear_significance_weight_test(Z,Y,W,a)
    WX = Z .* W.^(1/2);
    WY = Y .* W.^(1/2);
    P = least_squares_estimation(WX,WY)
    [r c] = size(Z)
    Y_ = MEAN(Y);
    
    SSR = sum(W.*(Z * P - Y_).^2) / (c - 1)
    SSE = sum(W.*(Y - Z * P).^2) / (r - c)
    F = SSR / SSE
    
    f = finv(1 - a,c - 1,r - c)
    if f >= F
        R = 0;
    else
        R = 1;
    end
end