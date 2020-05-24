%训练的数据必须是随机出现的,如果是按顺序生成数据需要多算几次，或则将数据复制多份参数训练。
function [W1 b1 W2 b2 E FITS] = back_propagation2(DATA,W1,b1,W2,b2,a,TDATA)
    tic
    [rows cols] = size(DATA);
    FITS = [];
    E_ = Inf;
    for i = 1:rows
        a0 = DATA(i,[1:cols-1]);
        a1 = logsig(W1 * a0 + b1);
        a2 = purelin(W2*a1 + b2);
        
        e = DATA(i,cols) - a2;
        s2 = -2*purelin_diff(a2)*e;
        s1 = logsig_diff(a1)*W2'*s2;
        
        W2 = W2 - a * s2 *a1';
        b2 = b2 - a * s2;
        W1 = W1 - a * s1 * a0;
        b1 = b1 - a * s1;
        
        E = validate(W1,b1,W2,b2,TDATA);
        if E < 0.001%误差很小，结束迭代。
            break;
        end
        if E_ < E
            ct = ct + 1;%误差出现连续增大，例如3次增大，终止迭代。
            if ct == 3 && E < 0.001
                'increase continuously'
                break;  
            end
        else
            ct = 0;
        end
        E_ = E;
    end
    
    for i = 1:rows
        a0 = DATA(i,[1:cols-1]);
        a1 = logsig(W1 * a0 + b1);
        a2 = purelin(W2*a1 + b2);
        
        FITS = [FITS;a0 a2];
    end
    toc
end

function R = logsig_diff(x)
    R = diag((1 - x).*x);
end

function R = purelin_diff(x)
    rows = size(x,1);
    R = eye(rows,rows);
end

function E = validate(W1,b1,W2,b2,TDATA)
    [rows cols] = size(TDATA);
    E = 0;
    for i = 1:rows
        a0 = TDATA(i,[1:cols-1]);
        a1 = logsig(W1 * a0 + b1);
        a2 = purelin(W2*a1 + b2);
        
        e = TDATA(i,cols) - a2;
        E = E + e.^2;
    end
    E = E/rows;
end