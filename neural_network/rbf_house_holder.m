function X = rbf_house_holder(A,b)
    [row col] = size(A);
    Q = eye(row,row);
    R = A;
    A_ = A;
    for j = 1:col
        x = A_(j:row,j);
        x_size = size(x,1);
        w = zeros(x_size,1);
        w(1) = norm(x);
        v = w - x;
        P = (v*v') / (v'*v);
        Hp = eye(x_size,x_size) - 2*P;
        Hj = eye(row,row);
        Hj(j:row,j:row) = Hp;
        
        R = Hj*R;
        A_ = R;
        Q = Q*Hj;
    end
    %反向替代，求解参数
    r_num = size(R,1);
    QB = Q'*b;
    QB = QB(1:col);
    X = zeros(col,1);
    for j = -col:-1
        X_ = QB(-j);
        for i = -col:(j-1)
            X_ = X_ - R(-j,-i)*X(-i);
        end
        X(-j) = X_ / R(-j,-j);
    end
end