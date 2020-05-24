function N = SSE(P,X,Y)
    N = sum((Y - X*P).^2)
end