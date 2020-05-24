function [C DATA] = silhouette_pam(CELLS)
    cols = size(CELLS,2);
    C = [];
    DATA = [];
    for i = 1:cols
        Mi = cell2mat(CELLS(i));
        C = [C;Mi];
        ci = size(Mi,1);
        BS = zeros(ci,1);
        B_num = 0;
        A = [];
        for j = 1:cols
            Mj = cell2mat(CELLS(j));
            ci = size(Mi,1);
            if i == j
                for k = 1:ci
                    a = euclidean(Mi,Mi(k,:))/ci;%到簇内到平均距离。
                    A = [A;a];
                end
            else
                cj = size(Mj,1);
                B_num = B_num + cj;
                for k = 1:ci
                    b = euclidean(Mj,Mi(k,:));%到簇外的总距离。
                    BS(k) = BS(k) + b;
                end
            end
        end
        B = BS/B_num;
        D = B - A;
        DM = max([A B],[],2);
        DM_ = D .* DM.^-1;
        DATA = [DATA;DM_];
    end
end

function d = euclidean(DATA,D)
    [cols rows] = size(DATA);
    DM = DATA - D;
    SUM = zeros(cols,1);
    for i = 1:rows
       SUM = SUM + DM(:,i).^2;
    end
    d = sum(sqrt(SUM));
end