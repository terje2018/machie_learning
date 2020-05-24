%评估 基于欧式距离的pcoa的效果
% 20%为效果很差 10%效果一般 5%为效果比较好 2。5%效果好 0%完美
function S = strain_CMDS(D,M)
    S1 = (D - M).^2;
    S2 = D.^2;
    
    SS1 = sum(sum(S1)');
    SS2 = sum(sum(S2)');
    
    S = sqrt(SS1/SS2);
end