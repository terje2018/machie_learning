%将x,y之间的关系通过corralation matrix表示出来，便于分析多重共线性
function A = corXY(X,Y)
    YX = [Y X];
    A= COR(YX);
end