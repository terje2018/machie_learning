%引入核方法的概念，将对距离的计算分离出来，便于使用不同的距离计算方式，解决各种场景问题。
%这里主要引入manhattan距离作为示范。
%程序中，一旦将kernel方法分离出来，那么可以不改变整体框架的情况下，
%选用不同的kernel method来获取不同的距离度量，从而得到不同的效果。
function [K T]= PAM_kernel(DATA,k_num,method_index)
    [rows cols] = size(DATA);
    k_index = ceil(rand(k_num,1) * rows);
    K = DATA(k_index,:);
    T = zeros(k_num,1);
    while 1
        KDATA = {};
        for k = 1:k_num
            KDATA = [KDATA {[]}];
        end
        
        for i = 1:rows
            D = [];
            for j = 1:k_num
                d = kernel_method(K(j,:),DATA(i,:),method_index);%由于这里是计算距离，所以分布式环境下，可以将K传给每一个block，然后将结果散列，汇总到下游，便于下游本地计算。
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
                [k t]= swap_dist(KD,method_index);
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
    end
end

function [k trace_value] = swap_trace(DATA,method_index)%这里如果是分布式环境，在将每个分组在本地完成计算。也就是说可以在本地把最优的点找出来。
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

function [k dist] = swap_dist(DATA,method_index)%使用距离来度量没有使用散度度量那么好，因为当出现数据比较分散，比较稀疏，这个时候距离度量效果不如散度。优点在于计算简单。不需要取构建矩阵，对于数据量较大的情况优势明显。
    k = [];
    dist = Inf;
    [r c] = size(DATA);
    for i = 1:r
        d = kernel_method(DATA,DATA(i,:),method_index);
        if d < dist
            k = DATA(i,:);
            dist = d;
        end
    end
end

function d = manhattan(x,y)
    d = sum(sum(abs(x - y)')');
end

function d = kernel_method(x,y,index)
    switch index
        case 1
            d = manhattan(x,y);
    end
end