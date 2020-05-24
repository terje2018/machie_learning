function [MAL_ MAL PM M R V] = native_bayes_online(DATA,TEST,LABELS)
    [rows cols] = size(DATA);
    lr = size(LABELS,1);
    PM = ones(LABELS,1); %拉普拉斯平滑
    M = ones(LABELS,cols-1);%拉普拉斯平滑
    COUNT = LABELS;
    for i = 1:rows
        COUNT = COUNT + 1;
        label = DATA(i,1);
        lg = (DATA(i,[2:cols]) == 1);
        switch label
            case 1
                PM(1,:) = PM(1,:) + 1;
                M(1,:) = M(1,:) + lg;
            case 2
                PM(2,:) = PM(2,:) + 1;
                M(2,:) = M(2,:) + lg;
            case 3
                PM(3,:) = PM(3,:) + 1;
                M(3,:) = M(3,:) + lg;
        end
    end
    
    PM / COUNT
    P = log(PM / COUNT); %先验，log化，防止出现连乘之后数字过小。
    ML = [];
    for i = 1:LABELS
        r = log(1);
        for j = 1:cols-1
            if TEST(j,1) == 0
                r = r + log(1 - M(i,j)/COUNT);
            else
                r = r + log(M(i,j)/COUNT);
            end
        end
        ML = [ML;r];
    end
    MAL = P + ML;%MAP
    MAL_ = exp(MAL);
    [R,V] = max(MAL);
end