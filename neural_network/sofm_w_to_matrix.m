%便于通过点图打印出来，于数据源做对比。
function M = sofm_w_to_matrix(W)
    [l w h n] = size(W);
    M = zeros(h*n,l);
    count = 1;
    for i = 1:h
        for j = 1:n
            for k = 1:l
                for q = 1:w
                    M(count,k) = W(k,q,i,j);
                end
            end
           count = count + 1;
        end
    end
end