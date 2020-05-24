%先通过bg dw lbq检测，如果存在autocorrelation做一次转换
%得到新矩阵之后，可以使用ols，相当于将模型转换成高斯-马尔可夫 回归模型
function A = cochrance_orcutt_procedure(X,Y)
    [r c] = size(X);
    p = autocorrelation_function(X,Y,1);
    %DW_test_value(X,Y);
    
    Yt = Y(1:r-1,:);
    Y2tr = Y(2:r,:);
    Y_CO = Y2tr - p*Yt;
    
    Xt = X(1:r-1,:);
    X2tr = X(2:r,:);
    X_CO = X2tr - p*Xt;
    
    %DW_test_value([ ones(r-1,1) X_CO(:,[2:c])],Y_CO);
    
    A = [[ ones(r-1,1) X_CO(:,[2:c])] Y_CO];
end