function R = spearman_test(n,rs,a)
    t = abs((sqrt(n - 2)*rs) / sqrt((1 - rs.^2)))
    ti = abs(tinv((1-a/2),n-2))
    if ti < t
        R =1
    else
        R = 0
    end
end