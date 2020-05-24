function CI = mutil_forecast(Z,Y,z0,a)
    [r c] = size(Z)
    [ry cy] = size(Y)
    P = []
    for i = 1:cy
        P = [P least_squares_estimation(Z,Y(:,i))]
    end
    
    C = zeros(cy,cy)
    for i = 1:cy
        for j = 1:cy
            C(i,j) = (Y(:,i) - (Z * P(:,i)))' * (Y(:,j) - (Z * P(:,j)))
        end
    end
    
    CI = []
    Pz = P'*z0
    [pzr pzc] = size(Pz)
    for i = 1:pzr
        z = 1+ z0' * inv(Z'*Z) * z0
        rcc = (r / (r - c)) * C(i,i)
        z0ZZz0 = sqrt(z*rcc)
        num = (cy * (r - c))/(r - c + 1 + cy)
        finv((1-a),cy,r - c + 1 - cy)
        f = sqrt(num * finv((1-a),cy,r - c + 1 - cy))
        
        CI = [CI; Pz(i) - f * z0ZZz0 Pz(i) + f * z0ZZz0]
    end
end