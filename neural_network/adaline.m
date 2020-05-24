function R = adaline(LABELS,DATA,a,V)
    [num colsd] = size(DATA);
    cols = size(LABELS,2);
    W = eye(colsd,cols)';
    b = ones(cols,1);
    for j = 1:500
        for i = 1:num
            alpha = W*DATA(i,:)';
            error = LABELS(i,:)' - alpha;
            W = W + 2*a*error*DATA(i,:);
            b = b + 2*a*error;
        end
    end

    W*V + b
    R = hardlims(W*V + b);
end