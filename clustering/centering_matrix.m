function J = centering_matrix(n)
    J = eye(n,n) - (1/n) * ones(n,n);
end