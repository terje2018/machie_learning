%回归参数的均方误差，可以对无偏估计判断，便于判定岭回归中的k对整体估计的影响
%K = 0为无偏估计，对比k不等于0的情况，判断是否采用岭回归的方式。
function N = regression_params_MSE(X,Y,k)
    [r c] = size(X);
    
    %这里是理论上计算的有偏估计，如果大于无偏估计，那么可以采取这种方式来处理。
    B = least_squares_estimation(X,Y)
    s2 = MSRes(X,Y,B) 
    %K = (c*s2)/(B'*B) 使用交叉验证来获取
    iv = inv(X'*X + k*eye(c,c))%ink = inv(Xk'*Xk)
    TR = trace(iv * X'*X * iv )
    mse1 = s2 * TR
    mse2 = (k.^2) * B'*(iv.^2) * B
    
    Br = inv(X'*X + k*eye(c,c)) * (X'*X) * B
    
    N = mse1 + mse2
end