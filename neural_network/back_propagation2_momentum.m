function [W1 b1 W2 b2 E] = back_propagation2_momentum(DATA,W1,b1,W2,b2,a,gama,TDATA)
    tic
    [rows cols] = size(DATA);
    E = Inf;
    for i = 1:rows
        a0 = DATA(i,[1:cols-1]);
        a1 = logsig(W1 * a0 + b1);
        a2 = purelin(W2*a1 + b2);
        
        e = DATA(i,cols) - a2;
        s2 = -2*purelin_diff(a2)*e;
        s1 = logsig_diff(a1)*W2'*s2;
        
        gamaW2 = (1 - gama) * W2;
        W2 = W2 - a * s2 *a1';
        W2 = gamaW2 + gama*W2;
        
        gamaB2 = (1 - gama) * b2;
        b2 = b2 - a * s2;
        b2 = gamaB2 + gama*b2;
        
        gamaW1 = (1 - gama) * W1;
        W1 = W1 - a * s1 * a0;
        W1 = gamaW1 + gama*W1;
        
        gamaB1 = (1 - gama) * b1;
        b1 = b1 - a * s1;
        b1 = gamaB1 + gama*b1;
        
        E_ = validate(W1,b1,W2,b2,TDATA);
        if E_ > E
            %break;
        end
        E = E_;
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
end