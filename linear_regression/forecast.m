%预测中不仅需要对回归函数做估算，还需要对Y值做估算
%由于Y=z0'Bata - sita,这里会引进一个误差sita，造成相对比于对回归函数做估算，置信区间更大
function CI = forecast(Z,Y,z0,a)
    [r c] = size(Z)
    P = least_squares_estimation(Z,Y)
    p = P' * z0
    t = abs(tinv((1-a/2),r-c))
    y_hat = Z*P
    s = (Y - y_hat)
    ss = sum(s.^2)/(r - c)
    sv = sqrt((1 + z0'*inv(Z'*Z)*z0)*ss)
    
    CI = [p - t*sv p + t*sv]
end