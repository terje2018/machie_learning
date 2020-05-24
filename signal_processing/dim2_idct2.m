%可以用于非方阵的情况。
function X = dim2_idct2(Y)
    [r c] = size(Y);
    CL = dct_c(r);
    CR = dct_c(c);
    X = CL'*Y*CR;
end