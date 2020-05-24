%使用chi2来鉴定，如果pcoa计算出来的值，与实际的值偏差比较明显，那么拒绝原假设
%A为 1 - 原始的相关系数矩阵，这里的的A通常由A = 1 - phi得到，phi为相关系数矩阵
%是否可以是similarity_coefficient需要待定。我觉得应该可以这样做。
function R = strain_CMDS_coefficient(A,M,a)
    n = size(A,1);
    df = (n.^2 - n)/2 - 1;
    
    A(triu(A)==0) = 1;
    M(triu(M)==0) = 1;
    S = 2*(log(A) - log(M));
    SSUM = sum(sum(S)')
    chi2p = chi2inv(1-a,df)
    
    if SSUM < chi2p
        R = 0;
    else
        R = 1;
    end
end