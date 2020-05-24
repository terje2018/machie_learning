function RS = regression_params_significance_weight_test(X,Y,W,a)
    WX = X .* W.^(1/2);
    WY = Y .* W.^(1/2);
    P = least_squares_estimation(WX,WY);
    [r c] = size(WX);
    d = inv(WX'*WX);
    [dr dc] = size(d);
    D = diag(d).^(1/2);
    SSE = sum(W.*(Y - X*P).^2);
    rou = sqrt(SSE/(r - c));
    
    RS = [];
    t = abs(tinv(a/2,r-c));
    for i = 1:dr
        ti = P(i) / (rou * D(i));
        ti_abs = abs(ti);
        if ti_abs >= t
            RS = [RS;ti 1];
        else
            RS = [RS;ti 0];
        end
    end
    
end