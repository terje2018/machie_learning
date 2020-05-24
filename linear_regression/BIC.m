function R = BIC(X,Y)
    [r c] = size(X);
    P = least_squares_estimation(X,Y);
    SSE = sum((Y - X*P).^2);
    
    R = r * log(SSE/r) + c * log(r);
end