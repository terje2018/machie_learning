function R = hopfield(W2,B,V,VM)
    cols = size(VM,2)
    FLAG = zeros(cols,1)
    
    while sum(FLAG) ~= 1
        A = satlins(W2 * V + B)
        if sum(A - (VM(:,1))) == 0
            FLAG(1) = 1
        end
        if sum(A - (VM(:,2))) == 0
            FLAG(2) = 1
        end
        V = A
    end
    R = FLAG
end