%可以用于非方阵的情况。
%添加高通滤波器功能.
%使用hilb，并且通过p值，使得高频数据加强，低频降低。
function X = dim2_idct4(Y,p)
    [r c] = size(Y);
    h = p*hilb(r);
    Y = Y ./ h;
    CL = dct_c(r);
    CR = dct_c(c);
    X = CL'*Y*CR;
end