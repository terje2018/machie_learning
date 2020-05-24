function C = dct_c(n)
    C = ones(n,n);
    for i = 1:n
        for j = 1:n
            C(i,j) = cos((i-1)*(2*j-1)*pi/(2*n));
        end
    end
    C = sqrt(2/n)*C;
    C(1,:) = C(1,:) / sqrt(2);
end