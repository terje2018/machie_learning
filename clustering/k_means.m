%n != 1,=1没有意义
%p 初始化质点的方式，0是将A平均分割，然后每一块的平均值作为质点，1是随机选取
function [M G] = k_means(A,p,n)
    [r c] = size(A);
    gn = floor(r/n);
    
    G = [];
    for i = 1:n %初始化 centroid ，初始化分组。
        if i == 1
            cell = A([1:gn],:);
            G = [{cell}];
        else
            if i == n %最后一组，将把剩下的都包括
                cell = A([(i - 1)*gn + 1 : r],:);
                G = [G;{cell}];
            else
                cell = A([(i - 1)*gn + 1 : i*gn],:);
                G = [G;{cell}];
            end
        end
    end
    M = k_means_seeds(A,p,n);%centroid seeds
  
    M_ = zeros(n,c);
    G_ = [];
    while ~isequal(M,M_) %实际工程当中，这里很可能无法相等，会变成在一个可接受的误差范围
        M_ = M;
        G_ = [];
        for i = 1:n%清空，初始化
            G_ = [G_;{zeros(1,c) + Inf}];
        end
        
        for i = 1:n
            CM = cell2mat(G(i));%拿出第一个group的sample与n个质点计算距离
            length = size(CM,1);
            for j = 1:length
                T = [];
                cm = CM(j,:);
                for k = 1:n %离那一个centroid点最近
                    d = norm( (cm - M(k,:))' );%欧式距离
                    T = [T;d];
                end
                %将样本与质点做比较，离那一个质点最近，就划归到那一组
                [m index] = min(T);
                CM_ = cell2mat(G_(index));
                if isequal(CM_,(zeros(1,c) + Inf))%如果是第一个
                    CM_ = cm;
                else
                    CM_ =[CM_;cm];
                end
                G_(index) = {CM_};
            end
        end
        %调整质点的位置
        G = G_;
        for i = 1:n
            CM = cell2mat(G(i));
            if size(CM,1) == 1 %只有一个sample，那么质点就是该样本
               M(i,:) = CM;
            else
               M(i,:) = mean(CM);
            end
        end
    end
    M;
    G;
end