%随着样本的增大，和bgtest结果越来越相近
function R = LBQ_test(X,Y,k)
    [r c] = size(X);
    S = 0;
    for i = 1:k
        t = autocorrelation_function(X,Y,i).^2;
        S = S + t/(r-i);
    end
    
    R = r*(r+2)*S;
end