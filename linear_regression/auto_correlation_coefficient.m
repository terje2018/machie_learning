function P = auto_correlation_coefficient(X,Y)
    P = 1 - DW_test_value(X,Y)/2;
end