%自变量选择，向后算法选择。
function IDS = variables_selecting_backward(X,Y,a)
    [r c] = size(X);
    INDEX = [1:c];
    XI = [INDEX;X];
    Xi = [XI];
    bic_ = BIC(X,Y)
    
    for i = 2:c
        [ri ci] = size(Xi);
        Xij = Xi;
        flag = 1;
        for j = 2:ci
            Xi_t = Xi;
            %id = Xi_t(1,j)
            Xi_t(:,j) = []; % 清除某一列
            st = mutil_linear_significance_test(Xi_t(2:r+1,:),Y,0.05);
            if st == 1
                bic = BIC(Xi_t(2:r+1,:),Y)
                if bic < bic_ %显著性提高了
                    Xij = Xi_t
                    bic_ = bic;
                    flag = 0;
                end
            end
        end
        if flag == 0
            Xi = Xij; %继续循环
        else
            break; %拟合度无法提高，所以终止
        end
    end
    bic_
    IDS = [Xi(1,:)]
end