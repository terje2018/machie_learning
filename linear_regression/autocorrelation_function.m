function R =autocorrelation_function(X,Y,k)
    [r c] = size(X);
    B = least_squares_estimation(X,Y);
    E = Y - X*B;
    ME = MEAN(E);
    VE = 0;
    for i = 1:r
       VE = VE +  (E(i) - ME).^2;
    end

    Et = E(k+1 : r);
    Et_1 = E(1 : r - k);
    MEt = MEAN(Et);
    MEt_1 = MEAN(Et_1);
    COVE = 0;
    for i = 1:r - k
        COVE = COVE + (Et(i) -  MEt) * (Et_1(i) -  MEt_1);
    end
    R = COVE/VE;
end