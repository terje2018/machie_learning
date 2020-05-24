%P训练的数据
%T训练数据对应的输出
%b1第一层的偏置量

%P_第一层brf的中心点
%PRS第二层的参数，主要是w和b
function [P_ PRS] = rbf_ols(P,T,b1,VALIDATE_SET_P,VALIDATE_SET_T)
    num = size(P,1);
    A = [];
    P_ = [];
    for i = 1:num
        D = [];
        for j = 1:num
            d = radbas(norm(P(i,:) - P(j,:))*b1);
            D = [D;d];
        end
        A = [A D];%每一列代表着 p到某个w的距离 如果分布式环境下，可以直接求出转置过后的矩阵
    end
    G = [A' ones(num,1)];%这里如果不是一个方阵，A需要转置，这样才能形成行大于列的矩阵，之后进行ols，或则可以直接ls求出来
    TABLE = array2table(G);
    O = [];
    
    %选出第一个点，参考rbf_schmidt中的思路
    for i = 1:num
        mi = TABLE(:,i).Variables;
        hi = (mi' * T) / (mi' * mi);
        oi = (hi.^2 * (mi' * mi)) / (T'*T);
        O = [O;oi]; 
    end
    [value index] = max(O);
    O = [];
    mj = TABLE(:,index).Variables;
    pindex = str2double(extractAfter(TABLE(:,index).Properties.VariableNames,"G"));
    P_ = [P_;P(pindex,:)];
    TABLE(:,index) = [];
    E_ = Inf
    
    count = 0;%如果连续出现n+1次迭代误差小于n次迭代，终止迭代。主要是防止出现某次迭代使得整体误差稍微变大，而后面误差又快速变小。
    flag = 0;
    for k = 1:num
        g_size = size(TABLE,2);
        for i = 1:g_size - 1
            mi = TABLE(:,i).Variables;
            r = (mj' * mi) / (mj' * mj);
            rm = r * mj;
            mi = mi - rm; %将mi中投影到mj的部分给去掉,从而形成正交
            TABLE(:,i).Variables = mi;
            
            hi = (mi' * T) / (mi' * mi);
            oi = ((hi.^2) * (mi' * mi)) / (T'*T);
            O = [O;oi];
        end
        [value index] = max(O);
        O = [];
        mj = TABLE(:,index).Variables;
        pindex = str2double(extractAfter(TABLE(:,index).Properties.VariableNames,"G"));
        P_ = [P_;P(pindex,:)]
        X2 = rbf_mse(P,T,b1,P_);
        E = test_validate(VALIDATE_SET_P,VALIDATE_SET_T,P_,b1,X2);
        PRS = X2;
        vpa(E_)
        vpa(E)
        if E < 0.01%误差很小，结束迭代。
            break;
        end
        if E_ < E
            count = count + 1;%误差出现连续增大，例如3次增大，终止迭代。
            if count == 3
                break;
            end
        else
            count = 0;
        end
        E_ = E;
        TABLE(:,index) = [];
    end
    
end

function X2 = rbf_mse(P,T,b1,P_)
    p_num = size(P_,1);
    num = size(P,1);
    
    A = [];
    for i = 1:num
       D = [];
       for j = 1:p_num
           d = radbas(norm(P(i,:) - P_(j,:))*b1);%训练数据到中心点的距离
           D = [D;d];
       end
       A = [A D];
    end
    G = [A' ones(num,1)];
    
    %X2 = inv(G'*G)*G'*T;
    X2 = rbf_schmidt(G,T)
    %X2 = rbf_house_holder(G,T);
    %E = sum((T - G*X).^2);
end

function E = test_validate(VALIDATE_SET_P,VALIDATE_SET_T,W1,b1,X2)
    num = size(VALIDATE_SET_P,1);
    p_num = size(W1,1);
    A = [];
    for i = 1:num
       D = [];
       for j = 1:p_num
           d = radbas(norm(VALIDATE_SET_P(i,:) - W1(j,:))*b1);%训练数据到中心点的距离
           D = [D;d];
       end
       A = [A D];
    end
    G = [A' ones(num,1)];
    Y = G * X2;
    E = sum((Y - VALIDATE_SET_T).^2)/num;
end
