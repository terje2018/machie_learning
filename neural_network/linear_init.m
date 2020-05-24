function W = linear_init(DATA,dimx,dimy)
    cols = size(DATA,2);
    m = mean(DATA);
    W = zeros(cols,1,dimx,dimy);
    M = COV(DATA);
    [v d] = eig(M);
    v2 = v(:,[1,2]);
    
    for i = 1:dimx
        for j = 1:dimy
            W(:,:,i,j) = m' + v2' * randn(2,1);
        end
    end
end