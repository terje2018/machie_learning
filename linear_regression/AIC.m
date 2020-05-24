function R = AIC(X,Y)
    [r c] = size(X);
    P = least_squares_estimation(X,Y);
    SSE = sum((Y - X*P).^2);
    
    R = r * log(SSE/r) + 2*c;
end