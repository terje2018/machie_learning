%局部加权会使得预测结果向数据相对聚集的地方偏移
%这里给出了调参k的一个方法，求残差平方和。
%NOTE:如果出现inv(x'*x)接近奇异说明残差平方会突然变大，所以对k的调整需要注意
function YHS = lwlr_msr(X,Y,k)
    [m n]= size(X);
    YH = zeros(m,1);
    for i = 1:m
        YH(i) = lwlr(X(i,:),X,Y,k);
    end
    YHS = sum((YH - Y).^2);
end