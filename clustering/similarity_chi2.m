function D = similarity_chi2(V1,V2)
    M =zeros(2,2);
    length = size(V1,1);
    for i = 1:length
        switch V1(i,1)
            case 1
                if V2(i,1) == 1
                    M(1,1) = M(1,1) + 1;
                else
                    M(1,2) = M(1,2) + 1;
                end
            case 0
                if V2(i,1) == 1
                    M(2,1) = M(2,1) + 1;
                else
                    M(2,2) = M(2,2) + 1;
                end
            otherwise
                'error';
                D = 0;
                return
        end
    end
    a = M(1,1);
    b = M(1,2);
    c = M(2,1);
    d = M(2,2);
    
    D = (a*d - b*c)/sqrt((a+b)*(c+d)*(a+c)*(b+d));
end