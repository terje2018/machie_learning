%可以用于非方阵的情况。
function Y = dim2_dct2(X)
    [r c] = size(X);
    CL = dct_c(r);
    CR = dct_c(c);
    Y = CL*X*CR';
end