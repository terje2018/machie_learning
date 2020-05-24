%cook distance判断是否是强影响点
function CDS = cook_distance(X,Y)
    [r c] = size(X)
    RS = T_inner_residual(X,Y)
    H = hat_matrix(X)
    HD = diag(H)
    
    HD_ = (ones(r,1) - HD).^(-1)
    HII = HD .* HD_
    
    p = (r - c)/(c + 1)
    B = (T_inner_residual(X,Y).^2)/(r - c)
    
    CDS = HII .* (p * B)
end