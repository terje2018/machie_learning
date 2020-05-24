function X = dim2_idct(Y)
    [r c] = size(Y);
    C = dct_c(r);
    X = C'*Y*C;
end