%p213
%https://deepnotes.io/softmax-crossentropy
function [WEIGHTS,B]= max_soft_stoc_grad_descent_cross_entropy4(DATA,WEIGHTS,B,a)
    [rows col]= size(DATA);
    LABELS = DATA(:,1);
    DATA_ = DATA(:,[2:col]);
    [r c] = size(WEIGHTS);
    for j = 1:rows
        L0 = logsig(WEIGHTS * DATA_(j,:)' + B);
        LB = label_convert(LABELS(j),r);
        
        S = (L0 - LB);%使用cross_entropy，这里可以把sigmod的导数给消除掉。从而简化了计算。
        W_ = S * DATA_(j,:);
        WEIGHTS = WEIGHTS - a*W_;
        B = B - a*S;
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
        L0 = logsig(WEIGHTS * DATA_(j,:)' + B);
        LB = label_convert(LABELS(j),r);
        
        s = sum((L0 - LB).^2);
        S = S + s;
        
        [v i] = max(L0);
        if i == LABELS(j)
            acc = acc + 1;
        end
    end
    C = acc;
end