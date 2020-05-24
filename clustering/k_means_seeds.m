%工程上需要判定A的column不能和n相当，两个相当时clutering的意义就消失了
%p 初始化质点的方式，0是将A平均分割，然后每一块的平均值作为质点，1是随机选取
function M = k_means_seeds(A,p,n)
    [r c] = size(A);
    gn = floor(r/n);
    switch p
        case 0
            M = zeros(n,c);%centroid
            for i = 1:n %初始化 centroid ，初始化分组。
                if i == 1
                    item = A([1:gn],:);
                    M(i,:) = mean(item);
                else
                    item = A([(i - 1)*gn + 1 : i*gn],:);
                    M(i,:) = mean(item);
                end
            end
        case 1
            M = zeros(n,c);%centroid
            ms = [];
            %随机挑选出质点，这里还可能存在一个问题，如果原始数据中存在大量相同的数据，这里的M可能出现相同。不过通常情况下不会出现。
            while size(ms,1) ~= n
                m = floor(rand(1)*r);
                if find(ms == m)
                else
                    if m ~= 0
                        ms = [ms;m];
                    end
                end
            end
            M = A(ms,:);    
        otherwise
            M = zeros(n,c);%centroid
            for i = 1:n %初始化 centroid ，初始化分组。
                if i == 1
                    item = A([1:gn],:);
                    M(i,:) = mean(item);
                else
                    item = A([(i - 1)*gn + 1 : i*gn],:);
                    M(i,:) = mean(item);
                end
            end
    end
end