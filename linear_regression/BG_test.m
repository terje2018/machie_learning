%自相关性检测，这里如果样本不大，并且dw检测的时候，得到的结果在边界附近，可能会得出现偏差
%注意观察辅助回归模型通过ols得到的回归系数，如果有比较明确的相关性的话，回归系数会有所表现。
function R = BG_test(X,Y,k)
    [r c] = size(X);
    B = least_squares_estimation(X,Y);
    E = Y - X*B;
    k_ = k+1;
    Et = E(k_ : r);
    Et_k = [];
    l = r - k;
    for i = 1:k
        Eti = E([i : l+i-1],:);
        Et_k = [Et_k Eti];
    end
    X_Et_k = [X([k_ : r],:) Et_k];
    r2 = R_square(X_Et_k, Et)
    T = r - k;
    R = T*r2;
    
end