function CS = error_point_checking_vars_base_deleting(X,Y)
    [r c] = size(X);
    P = least_squares_estimation(X,Y);
    R = T_inner_residual(X,Y);
    
    SSE = sum((Y - X * P).^2);
    C = SSE/(r - c);
    
    a = (r - c)/(r - c - 1);
    b = ((r - c - 1).^(-1)) * (R.^2);
    O = ones(r,1)
    CS = [(O .* C) ((O * a - O.*b) .* C)];
    
end