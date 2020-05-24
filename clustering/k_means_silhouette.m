% s(i) = [ b(i) - a(i) ] / max( a(i),b(i) )
% b为群中的点到其他群的最小距离
% a为群中的点到本群中的其他点的平均距离
% 度量每一个点在分群中的表现，整个可以概括出分群的轮廓
function S = k_means_silhouette(G)
    %MM = average_within_group_distance(G)
    %LMM = lowset_average_btw_group_distance(G)
    
    [MM LMM] = item_clustering_distance(G)
    M = LMM - MM
    MAXMM = max([MM LMM]')'
    S = M .* MAXMM.^-1
end

%求组里面每一个点到其他点的平均距离
function MM = average_within_group_distance(G)
    [gr gc] = size(G);
    MM = [];
    for j = 1:gr
        GM = cell2mat(G(j));
        [length wide]= size(GM);
        
        for k = 1:length
            DM = GM(k,:) .* ones(length,wide) - GM; %与群里面其他点的关系
            D = 0;
            if isequal(DM,zeros(length,wide))
                D = 0;
            else
                D = sum(sqrt(sum((DM .* DM)')')) / length; %计算与其他点的平均距离
            end
            MM = [MM;D];%某个点到同一个组里面的其他点的距离的平均值。
        end
    end
end

function LMM = lowset_average_btw_group_distance(G)
    [gr gc] = size(G);
    LMM = [];
    for i = 1:gr
        GM = cell2mat(G(i));
        length = size(GM,1);
        for j = 1:length
            VD = [];
            for k = 1:gr
                GM_ = cell2mat(G(k));
                [r_ c_] = size(GM_);
                DM = GM(j,:) .* ones(r_,c_) - GM_;% 某个点到其他组（群）里面的其他点。
                D = sum(sqrt(sum((DM .* DM)')')) / r_;%计算与其他群的点的平均距离
                VD = [VD;D];
            end
            VD(i) = [];%去掉自己所在的群
            LMM = [LMM;min(VD)];
        end
    end
end

function [MM LMM] = item_clustering_distance(G)
    [gr gc] = size(G);
    LMM = [];
    MM = [];
    for i = 1:gr
        GM = cell2mat(G(i));
        length = size(GM,1);
        for j = 1:length
            VD = [];
            for k = 1:gr
                GM_ = cell2mat(G(k));
                [r_ c_] = size(GM_);
                DM = GM(j,:) .* ones(r_,c_) - GM_;% 某个点到其他组（群）里面的其他点。
                D = sum(sqrt(sum((DM .* DM)')')) / r_;%计算与其他群的点的平均距离
                VD = [VD;D];
            end
            MM = [MM;VD(i)];%需要去掉的点，恰好是自己与所在的群平均距离
            VD(i) = [];%去掉自己所在的群
            LMM = [LMM;min(VD)];
        end
    end
    [MM LMM];
end