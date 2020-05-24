function R = hamming(W,V)
    rows = size(V,1)
    w_rows = size(W,1)
    B = ones(w_rows,1)*rows
    R_ = feedforward_layer(W,B,V)
    R = back_layer(W,R_)
end

function R = feedforward_layer(W,B,V)
    R = purelin(W*V + B)
end

function R = back_layer(W,V)
    rows = size(V,1)
    W2 = zeros(rows,rows)
    W2 = W2 - 1/4*(rows - 1)
    
    W2(logical(eye(rows))) = 1 %¶Ô½ÇÏß»»³É1
    
    while sum(V > 0) ~= 1
        V = poslin(W2*V)
    end
    R = V
end