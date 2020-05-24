function Y = dim2_dct(X)
    [r c] = size(X);
    C = dct_c(r);
    Y = C*X*C';
end