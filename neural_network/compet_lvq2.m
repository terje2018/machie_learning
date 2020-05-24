%使用lvq2算法实现clustering
%为了避免在子类之间出现"饿死"的情况，需要通过shrinkage收缩率和yield回避来调节。
%shrinkage在0～1之间，通常会大于0。5。过小的shrinkage意味着收缩速度快。
%yield回避，通过回避机制使得连续获胜的神经元能不断减少获胜的概率。从而让出机会给其他神经元，避免其他神经元饿死。
%yield根据实际情况来确认，数值越大，说明下一次越不容易获胜。
%
%如果shrinkage比较小，而yield比较大，虽然每个神经元都可能获胜，但是容易出现获胜神经元变动太快，影响结果。
%如果shrinkage比较大，而yield比较小，可能出现神经元饿死。
%所以shrinkage和yield，需要根据实际情况来确定。
function [W1 W2]= compet_lvq2(DATA,W1,W2,a,shrinkage,yield,TDATA)
    LABLES = DATA(:,1);
    [r c] = size(DATA);
    DATA_ = DATA(:,[2:c]);
    class_num = size(W2,1);
    subclass_num = size(W2,2)/class_num;
    B = zeros(class_num*subclass_num,1);
    
    %W2 = zeros(class_num,class_num*subclass_num);
    %for i = 1:class_num
    %    W2(i,[(i-1)*subclass_num+1 : i*subclass_num]) = 1;
    %end
    
    %W1 = randn(class_num*subclass_num,c-1);
    for i = 1:r
        L = compet(W1,DATA_(i,:));
        a2 = W2 * L;
        LB = label_convert(LABLES(i),class_num);
        if a2 == LB
            w1 = W1(logical(L),:);
            q = DATA_(i,:);
            W1(logical(L),:) = w1 + a * (q - w1);
        else
            [v ii] = max(LB);%每个subclass里面做竞争，算出最小值，w1往最近的地方移动。
            Ws = W1([((ii-1)*subclass_num +1) : ii*subclass_num],:);
            Bs = B([((ii-1)*subclass_num +1) : ii*subclass_num],1);
            [Ls Bs]= compet_in_subclass(Ws,DATA_(i,:),Bs,shrinkage,yield);
            %Ls = ones(subclass_num,1);
            w1 = Ws(logical(Ls),:);
            q = DATA_(i,:);
            %q = DATA_(i,:) .* ones(c-1,subclass_num);
            lg = [zeros((ii-1)*subclass_num,1);Ls];
            W1(logical(lg),:) = w1 + a * (q - w1);%错分，向正确数据点靠近。
            lgb = [zeros((ii-1)*subclass_num,1);ones(subclass_num,1)];
            B(logical(lgb),1) = Bs;
            
            w1 = W1(logical(L),:);
            q = DATA_(i,:);
            W1(logical(L),:) = w1 - a * (q - w1);%错分，远离数据点。
        end
    end
    
    acc = 0;
    [r c]= size(TDATA);
    TLABLES = TDATA(:,1);
    for i = 1:r
        L = compet(W1,TDATA(i,[2:c-1]));
        a2 = W2 * L;
        LB = label_convert(TLABLES(i),class_num);
        if a2 == LB
            acc = acc + 1;
        else
            %TDATA(i,:)
            %L = compet(W1,TDATA(i,[2:c-1]))
            %a2
        end
    end
    acc
    r
    acc/r
end

function L = compet(W1,d)
    [r c] = size(W1);
    L = [];
    for i = 1:r
        l = - norm(W1(i,:)' - d');
        L = [L;l];
    end
    [v,i] = max(L);
    L(:) = 0;
    L(i) = 1;
end

%添加偏移量b，防止出现饿死的情况。
%每次运算会对偏移量做收缩shrinkage，每次获胜的神经元需要增加一定的让步值yield，从而保证每个神经元都可以获胜。
function [L Bs]= compet_in_subclass(Ws,d,Bs,shrinkage,yield)
    [r c] = size(Ws);,
    L = [];
    for i = 1:r
        l = - norm(Ws(i,:)' - d') + Bs(i);
        L = [L;l];
    end
    [v,i] = max(L);
    L(:) = 0;
    L(i) = 1;
    Bs(:) = Bs(:) * shrinkage;
    Bs(i) = Bs(i) - yield;
end

function z = label_convert(label,r)
    z = zeros(r,1);
    z(label) = 1;
end