function W = dft_ic(n)
    W = ones(n,n);
    for ii = 1:n
        for j = 1:n
            W(ii,j) = (exp(i*2*pi/n)).^((ii-1)*(j-1));
        end
    end
    W = sqrt(1/n)*W;
end