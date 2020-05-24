%如果每一行为一个item，那么矩阵需要转置
function M = similarity_coefficient_matrix(A,n)
    [r c] = size(A);
    M = zeros(c,c);
    for i = 1:c
        for j = 1:c
            M(i,j) = similarity_coefficient(A(:,i),A(:,j),n);
        end
    end
end