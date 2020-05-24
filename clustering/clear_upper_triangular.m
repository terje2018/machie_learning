function Z = clear_upper_triangular(A,f)
    length = size(A,1);
    index = num2cell([1:length]);
    for i = 1:length
        for j = i:length;
            A(i,j) = f;
        end 
    end

    Z = A;
end