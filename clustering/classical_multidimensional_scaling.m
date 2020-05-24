function [Y E]= classical_multidimensional_scaling(EM,d)
    %EM = euclidean_distance_matrix(A)
    n = size(EM,1);
    J = centering_matrix(n);
    B = (-1/2) * J * (EM.^2) * J;
    
    [V D] = eig(B);
    
    VM = V(:,[1:d]);
    DM = D([1:d],[1:d]).^(1/2);
    E = diag(D);
    %sum(DD([1:d]))/sum(DD);
    %(V*sqrt(D)) *  (sqrt(D)*V')
    Y = VM*DM;
end