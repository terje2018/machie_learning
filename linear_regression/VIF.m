%VIF>5，存在多重共线性
function V = VIF(X)
    [r c] = size(X);
    V = []
    for i = 1:c
        Xj = X;
        Xi = X(:,i);
        Xj(:,i) = [];
        Xj = [ones(r,1) Xj];
        R = R_square(Xj,Xi);
        
        V = [V;i 1/(1 - R)];
    end
    V;
end