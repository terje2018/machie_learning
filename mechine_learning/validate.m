function R = validate(TDATA,WEIGHTS)
    count = 0;
    [r c]= size(TDATA);
    [row col] = size(WEIGHTS);
    for i = 1:r
        E = [];
        for j = 1:col
            e = exp(TDATA(i,[2 : c]) * WEIGHTS(:,j));
            E = [E;e];
        end
       [ele,index] = max(E);
       if index == TDATA(i,1)
           count = count + 1;
       end
    end
    R = count / r;
end