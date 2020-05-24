%估算残差的均方 residual mean square
%这里的z是加入了ones（x，1）,采用r - c。
%有些书上会写r，这里是的r是矩阵没有加ones的时候的维度，所以会n - r -1
function S = MSRes(Z,Y,P)
    [r c] = size(Z);
    y_hat = Z*P;
    s = (Y - y_hat);
    S = sum(s.^2)/(r - c);
end