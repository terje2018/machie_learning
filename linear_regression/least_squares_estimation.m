function P = least_squares_estimation(Z,Y)
    P = inv(Z'*Z)*Z'*Y;
end