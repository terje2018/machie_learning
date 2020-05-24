function [U S] = GMM_EM(DATA,W,num)
    [rows cols] = size(DATA);
    [r c] = size(W);
    index = floor(rows*rand(r,1));
    U = DATA(index,:)%期望
    S = [];%方差
    L_ = 0;
    for i = 1:r
        S = [S;eye(cols,cols)];
    end
    for k = 1:num
        %E step
        Eij = [];
        for i = 1:rows
            Ej = [];
            for j = 1:r
                e = [];
                if j == 1
                    e = W(j) * mvnpdf(DATA(i,:)',U(j,:)',S([1:j*cols],:));
                else
                    e = W(j) * mvnpdf(DATA(i,:)',U(j,:)',S([(j-1)*cols+1:j*cols],:));
                end
                Ej =[Ej,e];
            end
            Ej = Ej/sum(Ej);
            Eij = [Eij;Ej];
        end
        %M step
        N = sum(sum(Eij)');
        W = (sum(Eij) / N)';
        for i = 1:r
            SE = sum(Eij(:,i) .* DATA);
            SSE = sum(Eij(:,i));
            U(i,:) = SE .* SSE.^-1;

            SS = (Eij(:,i) .* (DATA - U(i,:)))' * (DATA - U(i,:));
            S([(i - 1) * cols + 1:i*cols],:) = SS .* SSE.^-1;
        end
        
        %W
        %U
        %S
        L = ML(DATA,W,U,S);
        if abs(L - L_) < 0.001
            'break'
            L
            break;
        else
            L_ = L;
        end
    end

end
%用作收敛
function L = ML(DATA,W,U,S)
    [rows cols] = size(DATA);
    [r c] = size(W);
    L = 0;
    for i = 1:rows
        Li = 0;
        for j = 1:r
            if j == 1
                e = W(j) * mvnpdf(DATA(i,:)',U(j,:)',S([1:j*cols],:));
            else
                e = W(j) * mvnpdf(DATA(i,:)',U(j,:)',S([(j-1)*cols+1:j*cols],:));
            end
            Li = Li + e;
        end
        L = L + log(Li);
    end
end