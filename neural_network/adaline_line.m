function R = adaline_line(LABELS,DATA,a,V)
    [num colsd] = size(DATA);
    cols = size(LABELS,2);
    W = zeros(colsd,cols)';
    %W = [0 0 0 0;0 0 0 0];
    for j = 1:500
        for i = 1:num
            alpha = W*DATA(i,:)';
            error = LABELS(i,:)' - alpha;
            W = W + 2*a*error*DATA(i,:);
        end
    end

    W*V
    R = hardlims(W*V);
end