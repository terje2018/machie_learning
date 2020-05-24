%Y = W*A*W将矩阵分解，使用dft_i获取到IW
%可以通过W'*Y*W = A，还原回去的。
%也通过IW*Y*IW' = A转换回去，不过这需要求一次变换矩阵的逆。
%通常不会专门去求，因为已经有一个矩阵W了，通过W'可以得到W的逆。
function W = dft_c(n)
    W = ones(n,n);
    for ii = 1:n
        for j = 1:n
            W(ii,j) = (exp(-i*2*pi/n)).^((ii-1)*(j-1));
        end
    end
    W = sqrt(1/n)*W;
end