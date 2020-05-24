%注意和ols的区别，hebb计算量太大，并且需要求pinv，使得分布式环境下实现难度大。
function INDEX = hebb(LABLES,DATA,V)
    W = LABLES*pinv(DATA);
    R_ = hardlims(W*V)
    
    distance = Inf;
    INDEX = Inf;
    n = size(LABLES,2);
    LR_ = R_ == 1;
    for i = 1:n
        LV = LABLES(:,i) == 1;
        hamming_distance = sum(xor(LV,LR_));
        if hamming_distance < distance
           distance = hamming_distance;
           INDEX = i;
        end
    end
end