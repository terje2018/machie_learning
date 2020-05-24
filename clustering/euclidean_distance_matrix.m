function M = euclidean_distance_matrix(A)
    [r c] = size(A);
    M = zeros(c,c);
    for i = 1:c
        for j = 1:c
            M(i,j) = euclidean_distance(A(:,i),A(:,j));
        end
    end
end