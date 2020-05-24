%对每一列的variable做分析，如果出现负值，说明1 0的出现相反，如果结果是-1，说明完全相反。
function M = similarity_chi2_matrix(A)
    [r c] = size(A);
    M = zeros(c,c);
    for i = 1:c
        for j = 1:c
            M(i,j) = similarity_chi2(A(:,i),A(:,j));
        end
    end
end