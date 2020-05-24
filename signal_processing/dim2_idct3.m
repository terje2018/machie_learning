%可以用于非方阵的情况。
%添加低通滤波器功能,把低于p的值给过滤掉，从而使得图片模糊。
function X = dim2_idct3(Y,p)
    lg = abs(Y) < p;
    Y(lg) = 0;
    [r c] = size(Y);
    CL = dct_c(r);
    CR = dct_c(c);
    X = CL'*Y*CR;
end