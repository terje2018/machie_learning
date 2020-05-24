%NOTE:每一次删除、添加一个自变量，都可能造成参数的显著性发生变化，理想状态下，需要不断的去判断
%对每一次的添加和删除都做参数显著性判断，并且对拟合度的影响，这样做有点麻烦。
%所以简化了步骤，直接删除自变量之后，对参数做显著性判断，然后选出拟合度最好的自变量作为结果
function IDS = variables_selecting_both(X,Y,a)
    [r c] = size(X);
    INDEX = [1:c];
    XI = [INDEX;X];
    Xi = [XI];
    ra_ = adjusted_RSquare(X,Y)%可以改成bic,效果会好一些
    
    while 1
        [ri ci] = size(Xi);
        Xij = Xi;
        flag = 1;
        for j = 2:ci
            Xi_t = Xi;
            %id = Xi_t(1,j)
            Xi_t(:,j) = []; % 清除某一列
            st = mutil_linear_significance_test(Xi_t(2:r+1,:),Y,0.05);
            if st == 1
                RS = regression_params_significance_test(Xi_t(2:r+1,:),Y,0.05)
                index = [];
                [rsr rsc] = size(RS);
                for i = 1:rsr
                    if RS(i,2) == 0
                        index = [index i];%去掉不显著的再算拟合程度
                    end
                end
                k = find(index == 1)
                if ~isempty(k)
                    index(1) = [];%第一列不能去掉
                end
                Xi_t(:,index) = []
                ra = adjusted_RSquare(Xi_t(2:r+1,:),Y)
                vpa(ra)
                if ra > ra_ %显著性提高了
                    Xij = Xi_t;
                    ra_ = ra;
                    flag = 0
                end
            end
        end

        if flag == 0
            Xi = Xij; %显著性提高了 继续循环
        else
            break; %拟合度无法提高，所以终止
        end
    end
    %ra_
    IDS = [Xi(1,:)]
end