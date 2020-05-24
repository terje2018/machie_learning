function R = test_estimate(Z,Y,a)
    [r c] = size(Z);
    R = [];
    for i = 1:r
        e = estimate(Z,Y,Z(i,:)',a);
        R = [R;(e(1)+e(2))/2 e abs(e(1)-e(2))];
    end
    
    R = [R Y];
end