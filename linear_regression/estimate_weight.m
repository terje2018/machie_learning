%这里暗含着z0通过某W.^1/2转换成wz0，之后才能符合模型的要求
%但是p = P' * z0如果加入w^1/2，那么获得的p也是含有w^1/2的
%最后sv1也是含有w^1/2，CI得出的结果就包含了w^1/2。
%由于最后需要将结果还原回去，这里的w^1/2又会被消除掉，所以算法可以写成如下
%但是实际上隐含了需要将预测参数加权才能符合模型，得到的结果要将加权去掉
%这个思想贯穿着整个回归，从线性回归到非线性
function CI = estimate_weight(Z,Y,W,z0,a)
    WX = Z .* W.^(1/2);
    WY = Y .* W.^(1/2);
    P = least_squares_estimation(WX,WY);
    [r c] = size(Z);
    p = P' * z0
    t = abs(tinv((1-a/2),r-c));
    y_hat = Z*P;
    s = (Y - y_hat);
    ss = sum(W .* s.^2)/(r - c);
    %(z0'*inv(Z'*Z)*z0);
    %z0'*inv(Z'*Z)*z0
    %z0'*inv(WX'*WX)*z0
    %sv = sqrt((z0'*inv(Z'*Z)*z0)*ss)
    sv1 = sqrt((z0'*inv(WX'*WX)*z0)*ss)
    
    CI = [p-t*sv1 p+t*sv1];
end