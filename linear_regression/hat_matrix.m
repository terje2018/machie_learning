function H = hat_matrix(X)
    H = X*inv(X'*X)*X';
end