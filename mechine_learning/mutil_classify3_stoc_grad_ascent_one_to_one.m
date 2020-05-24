function [WEIGHTS V]= mutil_classify3_stoc_grad_ascent_one_to_one(DATA,WEIGHTS,a,TDATA)
    [r c] = size(WEIGHTS);
    [row col] = size(DATA);
    %WEIGHTS_ = WEIGHTS;
    for i = 1:row
        label = DATA(i,1);
        data = DATA(i,[2:col]);
        switch label
            case 1
                error0 = 0 - logsig(data *  WEIGHTS(:,1));
                WEIGHTS(:,1) = WEIGHTS(:,1) + a * error0 * data';
                error1 = 1 - logsig(data *  WEIGHTS(:,3));
                WEIGHTS(:,3) = WEIGHTS(:,3) + a * error1 * data';
            case 2
                error0 = 0 - logsig(data *  WEIGHTS(:,2));
                WEIGHTS(:,2) = WEIGHTS(:,2) + a * error0 * data';
                error1 = 1 - logsig(data *  WEIGHTS(:,1));
                WEIGHTS(:,1) = WEIGHTS(:,1) + a * error1 * data';
            case 3
                error0 = 0 - logsig(data *  WEIGHTS(:,3));
                WEIGHTS(:,3) = WEIGHTS(:,3) + a * error0 * data';
                error1 = 1 - logsig(data *  WEIGHTS(:,2));
                WEIGHTS(:,2) = WEIGHTS(:,2) + a * error1 * data';
        end
    end
    
    V = zeros(r,1);
    for i = 1:r
        d = WEIGHTS(:,i)' * TDATA;
        if d < 0
           V(i) = V(i) + logsig(abs(d));
        else
            if i == r
                V(1) = V(1) + logsig(abs(d));
            else
                V(i + 1) = V(i + 1) + logsig(abs(d));
            end 
        end
    end
    VS = sum(exp(V));
    for i = 1:r
        V(i,2) = exp(V(i,1))/VS;
    end
    
    %{
    -WEIGHTS(1,1)/WEIGHTS(2,1)
    -WEIGHTS(3,1)/WEIGHTS(2,1)
    '----'
    -WEIGHTS(1,2)/WEIGHTS(2,2)
    -WEIGHTS(3,2)/WEIGHTS(2,2)
    '----'
    -WEIGHTS(1,3)/WEIGHTS(2,3)
    -WEIGHTS(3,3)/WEIGHTS(2,3)
    %}
end