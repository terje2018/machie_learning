%待议：这里是不是a/2？
function CI = linear_regression_confidence_interval(Z,Y,a)
    [r c] = size(Z);
    P = least_squares_estimation(Z,Y);
    
    S = MSRes(Z,Y,P)
    diag(inv(Z'*Z))
    D = diag(S*inv(Z'*Z));

    CI = [P - tinv(1 - a/2,r) * sqrt(D) P + tinv(1 - a/2,r) * sqrt(D)];
    CI = [CI abs(CI(:,1) - CI(:,2))]
end