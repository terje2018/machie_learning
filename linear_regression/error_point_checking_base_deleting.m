%分析异常点，判断是否是强影响点，或则异常点。
function EPS = error_point_checking_base_deleting(X,Y)
    PS = error_point_checking_params_base_deleting(X,Y);
    CS = error_point_checking_vars_base_deleting(X,Y);
    EPS = [PS CS]
end