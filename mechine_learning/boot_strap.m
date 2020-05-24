function A = boot_strap(DATA)
    rows = size(DATA,1);
    A = [];
    for i = 1:rows
        a = DATA(ceil(rand(1,1)*rows),:);
        A = [A;a];
    end
end