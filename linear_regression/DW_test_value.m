function V = DW_test_value(X,Y)
    P = least_squares_estimation(X,Y)
    E = Y - X*P;
    
    Et1 = E;
    Et1(1) = [];%去掉第一项
    
    Et0 = E;
    [r c] = size(Et0);
    Et0(r) = [];
    
    A = sum((Et1 - Et0).^2);
    B = sum(E.^2);
    
    V = A/B;
end