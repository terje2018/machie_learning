function [X XS]= conjugate_descent(F,MX,X0)
    XS =[]
    J = jacobian(F)';
    H = eval(jacobian(J));
    G0 = eval(subs(J,MX,X0));%初始位置
    P0 = -G0;%方向
    a0 = -(G0'*P0)/(P0'*H*P0)%学习率
    if isnan(a0)
       a0 = 0
    end
    X = X0 + a0*P0
    XS = [XS;X'];
    d = norm(X - X0);
    X0 = X;
    while d > 0.0001
        Gk = eval(subs(J,MX,X0));
        Pk = -Gk;
        bata = (Gk'*Gk)/(G0'*G0);
        Pkk = Pk + bata*P0;
        ak = -(Gk'*Pkk)/(Pkk'*H*Pkk);
        if isnan(ak)
            ak = 0
        end
        X = X0 + ak*Pkk
        XS = [XS;X'];
        d = norm(X - X0);
        G0 = Gk;
        P0 = Pkk;
        X0 = X;
    end
    X;
end