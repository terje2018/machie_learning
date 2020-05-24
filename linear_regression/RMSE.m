function N = RMSE(Z,Y,P)
    [r c] = size(Z);
    y_hat = Z*P;
    s = (Y - y_hat);
    N = sqrt(sum(s.^2)/r);
end