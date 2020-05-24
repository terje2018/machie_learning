function R = generalized_difference_4serial_correlation(X,Y)
    [r c] = size(Y);
    R1 = DW_test_value([ones(r,1) X(:,2)],Y);
    ps = serial_correlation_coefficient([ones(r,1) X(:,2)],Y)
    r1 = R_square([ones(r,1) X(:,2)],Y)
    X21 = (r-1)*r1
    p = serial_correlation_coefficient(X,Y);
    f1 = mutil_linear_F([ones(r,1) X(:,2)],Y)
    
    Yt1 = Y;
    Yt1(1) = [];
    Yt0 = Y;
    Yt0(r) = []; 
    Ystar = Yt1 - p*Yt0
    
    Xt1 = X;
    Xt1(1,:) = [] ;
    Xt0 = X;
    Xt0(r,:) = [];
    Xstar = Xt1 - p*Xt0
    size(Xstar);
    R2 = DW_test_value([ones(r-1,1) Xstar(:,2)],Ystar);
    pe = serial_correlation_coefficient([ones(r-1,1) Xstar(:,2)],Ystar)
    r2 = R_square([ones(r-1,1) Xstar(:,2)],Ystar)
    X22 = (r-2)*r2
    f2 = mutil_linear_F([ones(r-1,1) Xstar(:,2)],Ystar)
    R = [[ones(r-1,1) Xstar(:,2)] Ystar];
end