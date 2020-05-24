function D = similarity_coefficient(V1,V2,n)
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
    p = a + b + c + d;
    switch n
        case 1
            D = (a + d)/p;
        case 2
            D = 2*(a + d) / (2*(a + d) + b + c);
        case 3
            D = (a + d) / (a + d + 2*(b + c));
        case 4
            D = a / p;
        case 5
            D = a / (a + b + c);
        case 6
            D = 2*a / (2*a + b + c);
        case 7
            D = a / (a + 2*(b + c));
        case 8
            D =  a / (b + c);
        otherwise
            D = (a + d)/p;
    end
end