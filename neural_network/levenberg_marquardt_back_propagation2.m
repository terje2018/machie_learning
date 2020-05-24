%该方法容易出现无法收敛，初始化参数和lamda太难确认，一旦确认不好容易造成发散。
%即便使用其他方法训练出来的参数w b，带入到函数中，也会出现J中存在多重共线性，从而造成矩阵病态，使得无法求解。
%未来求出最少使用的神经元，可以采用如下方法：
%1，先使用bp算法，获取到参数 w b的数值。
%2，将w b带入到估算的数据中，例如需要估算 b3 - y + w21/(exp(- b1 - w11*x) + 1) + w22/(exp(- b2 - w12*x) + 1)
%当样本数据，w b都带入，形成矩阵A,使用eig(A'*A)，通过判断最大的特征值来决定神经元的个数。例如大于0.001的个数，
%通过该个数来判断需要多少个神经元，例如该个数有6个，mlp能逼近函数大于6个参数的最小神经元数量为7
function V = levenberg_marquardt_back_propagation2(DATA,W1,b1,W2,b2,num,lamda,e)
    [row cols] = size(DATA);
    S = [];
    J = [];
    V = [W1' b1' W2 b2]';
    R = [];
    E_ = Inf;
    for k = 1:num
        E = 0;
        R_ = [];
        J = [];
        for i = 1:row
            S_ = [];
            p = DATA(i,[1 : cols-1]);
            a0 = p;
            t = DATA(i,cols);
            n1 = W1*p + b1;
            a1 = logsig(n1);
            n2 = W2*a1 + b2;
            a2 = purelin(n2);
            E = E + (a2 - t).^2;
            R_ = [R_;a2 - t];

            S2_ = -purelin_diff(n2);
            %S_ = [S2_;S_];
            S1_ = logsig_diff(n1) * W2' * S2_;
            %S_ = [S1_;S_];
            %S = [S_ S];

            J1 = S1_ * a0;
            J2 = S1_;
            J3 = S2_ * a1;
            J4 = S2_;
            J = [J;J1' J2' J3' J4];
        end

        if E > e
            if E_ < E
                lamda = lamda * 10;
            else
                lamda = lamda / 10;
            end
            R = R_;
            inv(J'*J + lamda*diag(J'*J))
            inv(J'*J)
            Vk = -inv(J'*J + lamda*diag(J'*J).*eye(9,9))*J'*R;
            V = V + Vk;
            W1 = V([1:2],1);
            b1 = V([3:4],1);
            W2 = V([5:8],1)';
            b2 = V(9,1)';
            E_ = E;
        else
            E
            break;
        end
        
    end
    E

end

function R = logsig_diff(x)
    R = diag((1 - x).*x);
end

function R = purelin_diff(x)
    rows = size(x,1);
    R = eye(rows,rows);
end