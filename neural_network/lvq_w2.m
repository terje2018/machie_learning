function W2 = lvq_w2(class_num,subclass_num)
    W2 = zeros(class_num,class_num*subclass_num);
    for i = 1:class_num
        W2(i,[((i-1) * subclass_num + 1):i*subclass_num]) = 1;
    end
end