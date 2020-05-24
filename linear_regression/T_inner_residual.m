%学生化内残差，判定outlier
function RS = T_inner_residual(Z,Y)
    [r c] = size(Z);
    P = least_squares_estimation(Z,Y);
    E = Y - Z*P;
    H = Z*inv(Z'*Z)*Z';
    sum(E.^2)
    ss = sum(E.^2) / (r - c);
    
    hii = diag(H);
    [hr hc] = size(H);
    std = ss * (ones(hr,1) - hii);
    std = std.^(-1/2);
    
    RS = E .* std;
end