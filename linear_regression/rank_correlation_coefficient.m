%等级相关系数
%用于spearman检测
%NOTE:以下这种算法的步骤，在分布式环境下不可取，需要继续思考，如何在分布式环境下跑这个算法
function Rs = rank_correlation_coefficient(X,Y)
    [r c] = size(X);
    P = least_squares_estimation(X,Y)
    E = Y - X*P
    ABS_E = abs(E);
    ABS_E_RANK = sort(ABS_E);
    ABS_E_RANK_INN = [];
    
    for i = 1:r
        e = ABS_E(i,1);
        [col rw] = find(ABS_E_RANK(:,1) == e);
        ABS_E_RANK_INN = [ABS_E_RANK_INN;col(1,1)];
    end
    %残差于等级（rank）合并成一个矩阵
    ABS_E_RANK_2 = [ABS_E ABS_E_RANK_INN]
    
    rs_ = -inf
    for j = 1:c
        Xi = X(:,c)
        X_RANK = sort(Xi);
        
        RANK = []
        for i = 1:r
            xi = Xi(i,1);
            [col rw] = find(X_RANK(:,1) == xi);
            %RANK = [RANK;col(1,1)];%如果出现相同的元素，那么rank相同
            RANK = [RANK;mean(col)];%可能出现相同的值，那么取rank的均值
        end
        
        %将不同的自变量x于自身的rank合并
        Xi_RANK = [Xi RANK]
        %等级差平方和
        d2 = sum((Xi_RANK(:,2) - ABS_E_RANK_2(:,2)).^2)
        rs = 1 - 6/(r*(r*r - 1)) * d2
        
        if rs_ < rs
            rs_ = rs
        end
    end

    Rs = rs_
end