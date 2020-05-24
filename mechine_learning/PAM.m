function [K T KDATA_]= PAM(DATA,k_num)
    [rows cols] = size(DATA);
    k_index = ceil(rand(k_num,1) * rows);
    K = DATA(k_index,:);
    T = zeros(k_num,1);
    KDATA_ = {};
    while 1
        KDATA = {};
        for k = 1:k_num
            KDATA = [KDATA {[]}];
        end
        
        for i = 1:rows
            D = [];
            for j = 1:k_num
                d = euclidean(K(j,:),DATA(i,:));%由于这里是计算距离，所以分布式环境下，可以将K传给每一个block，然后将结果散列，汇总到下游，便于下游本地计算。
                D = [D;d];
            end
            [v in] = min(D);
            KD = cell2mat(KDATA(in));
            KD = [KD;DATA(i,:)];
            KDATA(in) = {KD};
        end
        K_ = zeros(k_num,cols);
        for i = 1:k_num
            KD = cell2mat(KDATA(i));
            [r c] = size(KD);
            for j = 1:r
                [k t]= swap_trace(KD); % !
                K_(i,:) = k;
                T(i,:) = t;
            end
        end
        
        %如果K点不在变化，终止迭代。
        logk = (K_ == K);
        logks = sum(sum(logk)');
        if logks == k_num*cols
            'break!'
            break;
        else
            K = K_;
        end
        KDATA_ = KDATA;
    end
end

function [k trace_value] = swap_trace(DATA)%这里如果是分布式环境，在将每个分组在本地完成计算。也就是说可以在本地把最优的点找出来。
    k = [];
    trace_value = Inf;
    [r c] = size(DATA);
    for i = 1:r
        D = DATA - DATA(i,:);
        DT = D' * D;
        tr = trace(DT);
        if tr < trace_value
            k = DATA(i,:);
            trace_value = tr;
        end
    end
end

function [k dist] = swap_dist(DATA)%使用距离来度量没有使用散度度量那么好，因为当出现数据比较分散，比较稀疏，这个时候距离度量效果不如散度。优点在于计算简单。不需要取构建矩阵，对于数据量较大的情况优势明显。
    k = [];
    dist = Inf;
    [r c] = size(DATA);
    for i = 1:r
        d = euclidean(DATA,DATA(i,:));
        if d < dist
            k = DATA(i,:);
            dist = d;
        end
    end
end

function d = euclidean(DATA,D)
    [cols rows] = size(DATA);
    DM = DATA - D;
    SUM = zeros(cols,1);
    for i = 1:rows
       SUM = SUM + DM(:,i).^2;
    end
    d = sum(sqrt(SUM));
end