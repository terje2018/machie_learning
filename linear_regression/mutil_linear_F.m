function F = mutil_linear_F(Z,Y)
    [r c] = size(Z);
    P = least_squares_estimation(Z,Y);
    Y_ = MEAN(Y);
    
    SSR = sum((Z * P - Y_).^2) / (c - 1);
    SSE = sum((Y - Z * P).^2) / (r - c);
    F = SSR / SSE;
end