%局部加权会使得预测结果向数据相对聚集的地方偏移
%这里只采用部分数据，从而可以提出一种思路：
%使用ols求出结果之后，与结果附近数据做加权，从而修正结果。
function YH = lwlr_part(tp,X,Y,k,range)
    XY = [X Y];
    [m n] = size(X);
    P = least_squares_estimation(X,Y);
    Yhat = tp*P;
    %做排序，以便将附近的数据用来做修正。
    XYS = sortrows(XY,n+1);
    lt = XYS(:,n+1) < Yhat + range;
    gt = XYS(:,n+1) > Yhat - range;
    logic = lt & gt;
    XYL = XY(logic,:);
    XL = XY(logic,[1:n]);
    YL = XY(logic,n+1);
    [lm ln] = size(XYL)
    XW = zeros(lm,ln-1);
    %w = inv(X'WX)*X'WY
    for i = 1:lm
        L = tp - XL(i,:);
        gs = exp(L*L'/(-2*(k.^2)));
        XW(i,:) = XL(i,:) .* gs;
    end
    w = inv(XW'*XL)*XW'*YL;
    YH = tp*w;
end