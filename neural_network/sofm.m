%d越大，特征数据点越容易聚合在一起，从而不利于数据的描述
%但是初期d越大越适合将特征数据点批量移动，便于描述数据。
%随着迭代大进行d会越来越小，这样便于微调
%a也是这样，一开始很大，随着迭代大进行越来越小。
%先降d，再降a。降d保证特征点可以分块分开，填充整个数据空间。d可以取0，仅仅关注一个神经元，不关注周边。
%降a相当于微调。
%
%
%不同于lvq2，sofm没有远离数据点的机制，所以本质上它有一个不断扩散到整个数据空间的趋势。
%通过对每个特征数据点周边数据点标签的统计，可以获得该数据点附近那一类数据比较多，从而通过标签实现数据可视化。
%这种可视化可以将高纬度的clustering数据，降到低维度来查看。
function [W U]= sofm(DATA,W,d,a)
    [r c] = size(DATA);
    [l w h q] = size(W);
    U = zeros(h,q);
    for k = 1:r
        p = DATA(k,[2,3]);
        [i j] = compet(W,p);
        [W U]= domain(W,U,i,j,DATA(k,:),d,a);
    end
end

function [i j] = compet(W,p)
    [l w h n] = size(W);
    M = zeros(h,n);
    for i = 1:h
        for j = 1:n
           M(i,j) = -norm(W(:,:,i,j) - p');
        end
    end
    [v i] = max(M);
    [vv j] = max(v');
     i = i(j);
end

function [W M]= domain(W,M,i,j,P,d,a)
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
            W(:,:,ii,ji) = W(:,:,ii,ji) + a*(P([2,3])' - W(:,:,ii,ji));
            M(ii,ji) = P(1);
        end
    end
    
end