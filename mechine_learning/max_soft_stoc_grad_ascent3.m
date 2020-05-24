function [WEIGHTS B] = max_soft_stoc_grad_ascent3(DATA,WEIGHTS,B,a)
    [rows col]= size(DATA);
    LABELS = DATA(:,1);
    DATA_ = DATA(:,[2:col]);
    [r c] = size(WEIGHTS);
    for k = 1:rows
        LB = label_convert(LABELS(k),r);
        SM = WEIGHTS * DATA_(k,:)' + B;
        [v index] = max(SM);
        sm_max = SM(index);
        sm = exp(SM - sm_max) / sum(exp(SM - sm_max));
        S = zeros(r,r);
        
        %计算soft max的导数。
        %这里和p168上使得导数对角化不一样，对角化是特殊情况。
        %真实的情况是，每个分量都可能影响导数。从而出现呢对角线两侧也有数据，而不是0。
        for j = 1:r
            for i = 1:r
                if i == j
                    S(j,i) =  sm(i) * (1 - sm(j));
                else
                    S(j,i) =  - sm(j) * sm(i);
                end
            end
        end

        %这里如果使用梯度下降，需要加上负号，如果是上升，不需要负号。
        %不伦上升还是下降，主要看推倒的时候如何确定，例如这里如果设定为 - Loss function，造成推到的过程中会出现符号被消除掉
        %消除掉负号，造成数学上梯度下降。如果没有消除负号，虽然也在做梯度下降，但是数学上是在做梯度抬升。
        e = (LB - sm);
        s = S * e;
        WEIGHTS = WEIGHTS + a * (s * DATA_(k,:));
        B = B + a * s;
    end
    
    [S C] = validate(DATA,WEIGHTS,B);
    S/rows
    C/rows
    
end

function z = label_convert(label,r)
    z = zeros(r,1);
    z(label) = 1;
end

function [S C]= validate(DATA,WEIGHTS,B)
    [rows col]= size(DATA);
    LABELS = DATA(:,1);
    DATA_ = DATA(:,[2:col]);
    [r c] = size(WEIGHTS);
    S = 0;
    acc = 0;
    for j = 1:rows
        SM = WEIGHTS * DATA_(j,:)' + B;
        [v i] = max(SM);
        sm_max = SM(i);
        sm = exp(SM - sm_max) / sum(exp(SM - sm_max));
        LB = label_convert(LABELS(j),r);
        
        s = sum((sm - LB).^2);
        S = S + s;
        
        [v i] = max(sm);
        if i == LABELS(j)
            acc = acc + 1;
        end
    end
    C = acc;
end