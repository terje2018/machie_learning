%度量分群的数量，如果结果减小的比较快，说明分群效果明显，如果减少的慢，说明增加分群效果不明显
function R = k_means_within_group_square(A,p,n)
    R = []
    for i = 1:n
        [M G] = k_means(A,p,i);
        D = within_group_distance(M,G);
        R = [R;i D];
    end
end

function D = within_group_distance(M,G)
    [r c] = size(M);
    [gr gc] = size(G);
    D = 0;
    for j = 1:gr
        GM = cell2mat(G(j));
        length = size(GM,1);
        for k = 1:length
            d = GM(k,:) - M(j,:);
            D = D + d*d';
        end
    end

end