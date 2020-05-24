function CI = estimate_autocorrelation(X,Y,z0,a,n)
    [r c] = size(X);
    R = LBQ_test(X,Y,1);
    chi2Sig = chi2inv(1-a,1);
    if R > chi2Sig %显著性检测chi2
      A = cochrance_orcutt_procedure(X,Y);
      X_CO = A(:,[1:c]);
      Y_CO = A(:,[c+1]);
      rou = autocorrelation_function(X,Y,1);
      %roun = rou*X(n,:)
      z = (z0' - rou*X(n,:))'%减去第n项，便于匹配模型
      
      B = least_squares_estimation(X_CO,Y_CO)
      %bz= B'*z
      %rouy = rou*Y(n,:)
      rst = B'*z + rou*Y(n,:)%估算值，需要加上第n个y
      
      
      t = abs(tinv((1-a/2),r-c));
      %估算残差的方差
      Y_CO_hat = X_CO*B;
      s2 = sum((Y_CO - Y_CO_hat).^2)/(r - c - 1);%残差平方和
      sv = sqrt((z'*inv(X_CO'*X_CO)*z)*s2)
      
      CI = [rst-t*sv rst+t*sv];
    else
        CI = estimate(X,Y,z0,a);
    end
    
end