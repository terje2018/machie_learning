%回归参数显著性检测，可以用来降维
function RS = regression_params_significance_test(Z,Y,a)
    P = least_squares_estimation(Z,Y);
    [r c] = size(Z);
    d = inv(Z'*Z);
    [dr dc] = size(d);
    D = diag(d).^(1/2);
    SSE = sum((Y - Z*P).^2);
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