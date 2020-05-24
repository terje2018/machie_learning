function X = rbf_schmidt(G,b)
    [row col] = size(G);
    R = zeros(col,col);
    Q = zeros(row,col);
    for j = 1:col
        y = G(:,j);
        for i = 1:j-1
            yi = Q(:,i);
            qi = yi/norm(yi);
            rij = qi'*y;
            R(i,j) = rij;
            y = y - rij*qi;
        end
        rjj = norm(y);
        R(j,j) = rjj;
        qj = y/rjj;
        Q(:,j) = qj;
    end
    %反向替代，求解参数
    r_num = size(R,1);
    QB = Q'*b;
    QB = QB(1:r_num);
    X = zeros(col,1);
    for j = -r_num:-1
        X_ = QB(-j);
        for i = -r_num:(j-1)
            X_ = X_ - R(-j,-i)*X(-i);
        end
        X(-j) = X_ / R(-j,-j);
    end
end