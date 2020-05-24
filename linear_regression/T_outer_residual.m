%%学生化外残差，判定outlier，没有内残差那么常用
function ES = T_outer_residual(Z,Y)
    [r c] = size(Z);
    P = least_squares_estimation(Z,Y);
    E = Y - Z*P;
    H = Z*inv(Z'*Z)*Z';
    hii = diag(H);
    ES = [];
    
    [er ec] = size(E);
    for i = 1:er
        SM = 0;
        for j = 1:er
            if i == j
                %do nothing
                continue
            end
            SM = SM + (Y(j,:) - Z(j,:)*P).^2;
        end
        ss_out = SM / (r - c -1);
        e = E(i)/ sqrt(ss_out * (1 - hii(i)));
        ES = [ES;e];
    end

end