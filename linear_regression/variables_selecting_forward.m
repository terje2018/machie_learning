%自变量选择，向前算法选择。
%当选出拟合度最高的自变量组合的时候，判断参数的显著性，去掉不显著的参数，使用剩下的
%也可以作为一种挑选自变量组合的简化方式
function IDS = variables_selecting_forward(X,Y,a)
    [r c] = size(X);
    INDEX = [1:c];%记录自变量的index
    IDS = [1];
    XI = [INDEX;X];
    Xi = [XI(:,1)];%初始化
    Xj = [XI(:,2:c)];
    bic_ = 0;%调整决定系数，如果下一次比上一次的系数大，循环终止
    for i = 2:c
        Fj = [];
        for j = 1:c - i + 1
            Xj_t = [Xi(2:r+1,:) Xj(2:r+1,j)];
            st = mutil_linear_significance_test(Xj_t,Y,a);
            if st == 1
                bic = adjusted_RSquare(Xj_t,Y);
                Fj = [Fj;bic Xj(1,j)];%将索引放入，便于定位
            end
        end
        SR = sortrows(Fj,1,"des")%排序，便于选出最显著的
        if isempty(SR) || SR(1,1) <= bic_
            %sr11 = SR(1,1)
            %ra_11 =  ra_
            break;%拟合程度下降或则不显著，造成无元素存在，停止添加自变量
        else
            bic_ = SR(1,1)%拟合程度有上升，继续添加自变量
        end
        
        IDS = [IDS SR(1,2)]%将列的索引记录下来
        %调整Xi,Xj
        XI_ = [];
        XJ_ = [];
        for i = 1:c
            k = find(IDS == i);
            if ~isempty(k)
                XI_ = [XI_ XI(:,i)];
            else
                XJ_ = [XJ_ XI(:,i)];
            end
        end
        Xi = XI_;
        Xj = XJ_;
    end
end