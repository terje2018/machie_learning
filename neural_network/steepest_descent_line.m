%沿直线最小化。用于二次函数
function X = steepest_descent_line(F,MX,X0)
    J = jacobian(F)';
    H = eval(jacobian(J));
    G = eval(subs(J,MX,X0));%初始位置
    P0 = -G;%方向
    a = -(G'*P0)/(P0'*H*P0)%学习率
    if isnan(a)
       a = 0
    end
    X = X0 - a*G
    d = norm(X - X0);
    X0 = X;
    while d > 0.000001
        Gk = eval(subs(J,MX,X0));
        Pk = -Gk
        a = -(Gk'*Pk)/(Pk'*H*Pk)
        if isnan(a)
            a = 0
        end
        X = X0 - a*Gk
        d = norm(X - X0);
        X0 = X;
    end
    X;
end