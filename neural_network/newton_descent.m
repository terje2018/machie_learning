function [X XS]= newton_descent(F,MX,X0)
    XS =[X0']
    J = jacobian(F)';
    H = jacobian(J);
    HG = eval(subs(H,MX,X0));
    G = eval(subs(J,MX,X0));%³õÊ¼Î»ÖÃ
    X = X0 - inv(HG)*G;
    XS = [XS;X'];
    d = norm(X - X0);
    X0 = X;
    while d > 0.0001
        HG = eval(subs(H,MX,X0));
        G = eval(subs(J,MX,X0));
        X = X0 - inv(HG)*G;
        XS = [XS;X'];
        d = norm(X - X0);
        X0 = X;
    end
    X
end