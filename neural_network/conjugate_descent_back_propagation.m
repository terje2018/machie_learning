function [W1 b1 W2 b2 E] = conjugate_descent_back_propagation(DATA,W1,b1,W2,b2,TDATA,num)
    [rows cols] = size(DATA);
    
    s2a2vag_g0 = zeros(1,size(W2,2));
    s2vag_g0 = zeros(1,1);
    s1evag_g0 = zeros(size(W1,1),1);
    s1vag_g0 = zeros(size(W1,1),1);
    
    s2a2vag_p0 = zeros(1,size(W2,2));
    s2vag_p0 = zeros(1,1);
    s1evag_p0 = zeros(size(W1,1),1);
    s1vag_p0 = zeros(size(W1,1),1);
    E_ = Inf;
    for k = 1:num
        s2a1sum = ones(1,size(W2,1));
        s2sum = ones(1,1);
        s1esum = ones(size(W1,1),1);
        s1sum = ones(size(W1,1),1);
        for i = 1:rows
            a0 = DATA(i,[1:cols-1]);
            a1 = logsig(W1 * a0 + b1);
            a2 = purelin(W2*a1 + b2);

            e = DATA(i,cols) - a2;
            s2 = -2*purelin_diff(a2)*e;
            s1 = logsig_diff(a1)*W2'*s2;

            s2a1sum = s2a1sum + s2*a1';
            s2sum = s2sum + s2;
            s1esum = s1esum + s1*e';
            s1sum = s1sum + s1;
        end
        s2a2vag_g = -s2a1sum / rows;
        s2vag_g = -s2sum / rows;
        s1evag_g = -s1esum / rows;
        s1vag_g = -s1sum / rows;

        if k == 1 %第一次搜索，采用最速下降法。
            a = line_minimized(DATA,W1,b1,W2,b2,s1evag_g,s1vag_g,s2a2vag_g,s2vag_g,0.075,100,0.01)
            W2 = W2 + a*s2a2vag_g;
            b2 = b2 + a*s2vag_g;
            W1 = W1 + a*s1evag_g;
            b1 = b1 + a*s1vag_g;
            
            s2a2vag_p0 = s2a2vag_g;
            s2vag_p0 = s2vag_g;
            s1evag_p0 = s1evag_g;
            s1vag_p0 = s1vag_g;
        else
            s2a2vag_dc_beta = (s2a2vag_g * s2a2vag_g') / (s2a2vag_g0 * s2a2vag_g0')
            s2vag_dc_beta = (s2vag_g' * s2vag_g) / (s2vag_g0' * s2vag_g0)
            s1evag_dc_beta = (s1evag_g' * s1evag_g) / (s1evag_g0' * s1evag_g0)
            s1vag_dc_beta = (s1vag_g' * s1vag_g) / (s1vag_g0' * s1vag_g0)
            
            s2a2vag_dc_p = s2a2vag_g + s2a2vag_dc_beta*s2a2vag_p0
            s2vag_dc_p = s2vag_g + s2vag_dc_beta*s2vag_p0
            s1evag_dc_p = s1evag_g + s1evag_dc_beta*s1evag_p0
            s1vag_dc_p = s1vag_g + s1vag_dc_beta * s1vag_p0
            
            a = line_minimized(DATA,W1,b1,W2,b2,s1evag_dc_p,s1vag_dc_p,s2a2vag_dc_p,s2vag_dc_p,0.075,100,0.01)
            W2 = W2 + a*s2a2vag_dc_p;
            b2 = b2 + a*s2vag_dc_p;
            W1 = W1 + a*s1evag_dc_p;
            b1 = b1 + a*s1vag_dc_p;
            
            s2a2vag_p0 = s2a2vag_dc_p;
            s2vag_p0 = s2vag_dc_p;
            s1evag_p0 = s1evag_dc_p;
            s1vag_p0 = s1vag_dc_p;
            
        end
        s2a2vag_g0 = s2a2vag_g;
        s2vag_g0 = s2vag_g;
        s1evag_g0 = s1evag_g;
        s1vag_g0 = s1vag_g;
        
        %s2a2vag_p0 = s2a1sum / rows;
        %s2vag_p0 = s2sum / rows;
        %s1evag_p0 = s1esum / rows;
        %s1vag_p0 = s1sum / rows;
       
        E = validate(W1,b1,W2,b2,TDATA)
        if E < 0.01%误差很小，结束迭代。
            break;
        end
        if E_ < E
            ct = ct + 1;%误差出现连续增大，例如3次增大，终止迭代。
            if ct == 3 && E < 0.1
                break;  
            end
        else
            ct = 0;
        end
            E_ = E;
    end
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

function a = line_minimized(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,eps,num,error_interval)
    [rows cols]= size(DATA)
    
    E_ = 0;
    for i = 1:rows
        a0 = DATA(i,[1:cols-1]);
        a1 = logsig(W1 * a0 + b1);
        a2 = purelin(W2*a1 + b2);

        e = DATA(i,cols) - a2;
        E_ = E_ + e.^2;
    end

    flag = 0
    k_ = 1/2;
    ak = 0
    bk = 0
    for k = 0:num%锁定区域
        k_ = k_ * 2;
        E = 0;
        W1 = W1 + k_*eps*pw1;
        b1 = b1 + k_*eps*pb1;
        W2 = W2 + k_*eps*pw2;
        b2 = b2 + k_*eps*pb2;
        for i = 1:rows
            a0 = DATA(i,[1:cols-1]);
            a1 = logsig(W1 * a0 + b1);
            a2 = purelin(W2*a1 + b2);

            e = DATA(i,cols) - a2;
            E = E + e.^2;
        end
        if E_ < E
            flag = flag + 1;
            E_ = E;
        else
            flag = 0;
            E_ = E;
        end
        if flag == 1
            if k == 0
               ak = 0;
               bk = 1 * eps;
            else
               ak = floor(k_ / 4) * eps;
               bk = k_ * eps;
               break;
            end
        end
    end
    
    r = 0.618;
    c = ak + (1-r)*(bk-ak);
    Fc = line_minimized_error(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,c)
    d = bk - (1-r)*(bk-ak);
    Fd = line_minimized_error(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,d)
    if abs(Fc - Fd) <= error_interval
        a = k*eps;
        return
    end
    for k = 1:num
        if Fc > Fd
            ak = c;
            bk = bk;
            c = ak + (1-r)*(bk-ak);
            Fc = line_minimized_error(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,c)
            d = bk - (1-r)*(bk-ak);
            Fd = line_minimized_error(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,d)
        else
            ak = ak;
            bk = d;
            d = bk - (1-r)*(bk-ak);
            Fd = line_minimized_error(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,d)
            c = ak + (1-r)*(bk-ak);
            Fc = line_minimized_error(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,c)
        end
        if abs(Fc - Fd) <= error_interval
            a = (ak + bk)/2
            break;
        end
    end

end

function E = line_minimized_error(DATA,W1,b1,W2,b2,pw1,pb1,pw2,pb2,eps)
    [rows cols] = size(DATA);
    W1 = W1 + eps*pw1;
    b1 = b1 + eps*pb1;
    W2 = W2 + eps*pw2;
    b2 = b2 + eps*pb2;
    E = 0;
    for i = 1:rows
        a0 = DATA(i,[1:cols-1]);
        a1 = logsig(W1 * a0 + b1);
        a2 = purelin(W2*a1 + b2);

        e = DATA(i,cols) - a2;
        E = E + e.^2;
    end
    E;
end

