%batch的算法比非batch稳定性更好
%非batch容易因为数据出现倾斜而造成特征数据分别也倾斜。
%例如如果数据集中虽然每一类数量都相等，但是出现的时候并不是完全随机出现，
%那么出现的顺序会改变特征数据的分布，从而造成不稳定的出现。
%
%d值，神经元附近的其他神经元都会被更新的范围。
%一开始d需要设置得大一些例如3，便于确定神经元负责的区域
%随着迭代的进行，逐步减少d值，直到0。
%w_linear = linear_init(SMAcp3(:,[2 3]),10,10);
%[w u]= sofm_batch([SMAcp3(:,[1 2 3])],w_linear,3,600);
%[w u]= sofm_batch([SMAcp3(:,[1 2 3])],w,2,600);
%[w u]= sofm_batch([SMAcp3(:,[1 2 3])],w,1,600);
%[w u]= sofm_batch([SMAcp3(:,[1 2 3])],w,0,600);
%
%
%初始化对结果影响很明显，batch通常采用linear init。如果是实时的算法，可以先做一个数据采样
%通过采样数据确定初始化W。
function [W U Z]= sofm_batch(DATA,W,d,n)
    [r c] = size(DATA);
    [l w h q] = size(W);
    for k = 1:n
        W_TEMP = zeros(l,w,h,q);
        N = zeros(h,q);
        [W_TEMP N] = compet(W,W_TEMP,N,DATA,d);

        for i = 1:h
            for j = 1:q
                if N(i,j) == 0
                     W_TEMP(:,:,i,j) = W(:,:,i,j);%保持原样。
                else
                     W_TEMP(:,:,i,j) = W_TEMP(:,:,i,j) / N(i,j);
                end
            end
        end
        W = W + (W_TEMP - W);
        
    end
    [U Z]= u_matrix(W,DATA);
end

function [W N U]= compet(W1,W,N,DATA,d)
    [l w h n] = size(W1);
    [r c] = size(DATA);
    M = zeros(h,n);
    %W = zeros(l,w,h,n);
    for k = 1:r
        P = DATA(k,[2,3])';
        for p = 1:h
            for q = 1:n
                l = - norm(W1(:,:,p,q) - P);
                M(p,q) = l;
            end
        end
        [v i] = max(M);
        [vv j] = max(v');
        i = i(j);
        [W N]=domain(W,N,i,j,P,d);
    end
end

function [W N]= domain(W,N,i,j,P,d)
    [x y r c] = size(W);
    id = i - d;
    if id <= 0
        id = 1;
    end
    jd = j - d;
    if jd <= 0
        jd = 1;
    end
    id_ = i+d;
    if i + d >= r
        id_ = r;
    end
    jd_ = j+d;
    if j + d >= c
        jd_ = c;
    end
    
    for ii = id:id_
        for ji = jd:jd_
            W(:,:,ii,ji) = W(:,:,ii,ji) + P;
            N(ii,ji) = N(ii,ji) + 1;
        end
    end
end


function [U Z]= u_matrix(W,DATA)
    [l w h n] = size(W);
    [r c] = size(DATA);
    U = zeros(h,n);
    Z = zeros(l,w,h,n);
    for p = 1:h
        for q = 1:n
            len = -Inf;
            for i = 1:r
               P = DATA(i,[2,3])';
               l =  -norm(W(:,:,p,q) - P);
               if l > len
                   U(p,q) = DATA(i,1);
                   len = l;
                   Z(:,:,p,q) = P;
               end
            end
        end
    end
end