%从整体角度来看，去掉ei之后，SSE的变化，如果变化比较大，进一步分析
function SSEi_ = error_point_checking_SSE_base_deleting(X,Y,i)
    P = least_squares_estimation(X,Y);
    SSE = sum((Y - X*P).^2);
    H = hat_matrix(X);
    ei2 = Y(i,:) - X(i,:)*P;
    hii = H(i,i);
    
    SSEi = SSE - ( ei2/(1 - hii) )
    SSEi_ = [SSE SSEi]
end