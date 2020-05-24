%k1 = unifrnd(-sqrt(6/((1+6)*5.^2)),sqrt(6/((1+6)*5.^2)),5,5,6);
%k2 = unifrnd(-sqrt(6/((6+12)*5.^2)),sqrt(6/((6+12)*5.^2)),5,5,6,12);
%这个版本比较原始，仅仅是将tanh换成了logsid，第一个卷积层有6个filter，第二个卷积层12个
%fc层没有使用rbf，而是使用了soft max
%pooling没有使用max pooling，而是使用avg pooling。
%这种网络架构可以将正确率提升到0.8。显然需要改进架构。
function [k1,B1,k2,B2,W,B3] = LeNet5_ConvNet_original(mnist,k1,k2,W,a)
    TRAINING = mnist.training;
    TR_LABELS = TRAINING.labels;
    TEST = mnist.test;
    T_LABELS = TEST.labels;
    B1 = zeros(1,1,6);
    B2 = zeros(1,1,12);
    B3 = zeros(10,1,1);
    
    for i = 1:60000
        %feedforward
        R1 = conv_C1(TRAINING.images(:,:,i),k1);
        C1 = logsig(R1 + B1);
        S1 = avg_pooling(C1);
        R2 = conv_C2(S1,k2);
        C2 = logsig(R2 + B2);
        S2 = avg_pooling(C2);
        f = vctz_concat(S2);
        y_hat = logsig(W*f + B3);
        y = label_vctz(TR_LABELS(i));

        %Backpropagation
        DT_Y = (y_hat - y) .* (y_hat' * (1 - y_hat));
        DT_W = DT_Y*f';
        DT_B = DT_Y;
        DT_f = W' * DT_Y;
        DT_S2 = reverse_vctz_concat(DT_f);%!
        DT_C2 = reverse_avg_pooling(DT_S2);
        DT_C2qs = DT_C2q_sigma(DT_C2,C2);
        DT_K2pq = K2pq(S1,DT_C2qs);
        DT_B2q = b2q(DT_C2,C2);
        DT_S1_p = S1p(DT_C2qs,k2);
        DT_C1p = reverse_avg_pooling(DT_S1_p);
        DT_C1ps = C1ps(DT_C1p,C1);
        DT_k11p = K11p(TRAINING.images(:,:,1),DT_C1ps);
        DT_b1p = b1p(DT_C1ps);
        
        BB1 = B1(1,1,:);
        BB11 = BB1(:);
        BB2 = B2(1,1,:);
        BB22 = BB2(:);
        
        k1 = k1 - a*DT_k11p;
        BB11 = BB11 - a*DT_b1p;
        k2 = k2 - a*DT_K2pq;
        BB22 = BB22 - a*DT_B2q;
        W = W - a*DT_W;
        B3 = B3 - a*DT_B;
        
        B1(1,1,:) = BB11;
        B2(1,1,:) = BB22;
    end
    
    count = 0;
    for i = 1:10000
        R1 = conv_C1(TEST.images(:,:,i),k1);
        C1 = logsig(R1 + B1);
        S1 = avg_pooling(C1);
        R2 = conv_C2(S1,k2);
        C2 = logsig(R2 + B2);
        S2 = avg_pooling(C2);
        f = vctz_concat(S2);
        y_hat = logsig(W*f + B3);
        [v ii] = max(y_hat);
        if (ii - 1) == T_LABELS(i)
            count = count + 1;
        end
    end
    count/10000
end

function B1 = b(l,w,h)
    B1 = zeros(l,w,h);
end

function R = conv_C1(I,k1)
    [fl fw fh fn] = size(k1);
    R = zeros(24,24,6);
    for p = 1:fh
        R(:,:,p) = convn(I,rot180(k1(:,:,p)),'valid');
    end
end

function R = conv_C2(S1,k2)
    [sl sw sh] = size(S1);
    [fl fw fh fn] = size(k2);
    R = zeros(sl-fl+1,sw-fw+1,fn);
    for k = 1:fn
        R_ = zeros(sl-fl+1,sw-fw+1,1);
        for n = 1:fh
            R_ = R_ + convn(S1(:,:,n),rot180(k2(:,:,n,k)),'valid');
        end
        R(:,:,k) = R_;
    end
end

function P = avg_pooling(C)
    [cl cw ch] = size(C);
    
    P = zeros(cl/2,cw/2,ch);
    for k = 1:ch
        for i = 1:(cl/2)
            for j = 1:(cw/2)
                c1 = C(2*i-1, 2*j-1, k);
                c2 = C(2*i-1, 2*j, k);
                c3 = C(2*i, 2*j-1, k);
                c4 = C(2*i, 2*j, k);
                a = (c1 + c2 +c3 +c4)/4;
                P(i,j,k) = a;
            end
        end
    end
end

function R = vctz_concat(S2)
    R = [];
    h = size(S2,3);
    for i = 1:h
        s2 = S2(:,:,i);
        R = [R;s2(:)];
    end
end

function LV = label_vctz(n)
    LV = zeros(10,1);
    LV(n+1) = 1;
end

function S = reverse_vctz_concat(f)
    S = zeros(4,4,12);
    for i = 0:11
        S([1:4],1,i+1) = f([i*16+1:i*16+4]);
        S([1:4],2,i+1) = f([i*16+5:i*16+8]);
        S([1:4],3,i+1) = f([i*16+9:i*16+12]);
        S([1:4],4,i+1) = f([i*16+13:i*16+16]);
    end
end

function C = reverse_avg_pooling(S)%！不取1/4
    [l w h] = size(S);
    C = zeros(l*2,w*2,h);
    for k = 1:h
        for i = 1:l
            for j = 1:w
                C((i-1)*2+1,(j-1)*2+1,k) = S(i,j,k);
                C((i-1)*2+1,(j-1)*2+1+1,k) = S(i,j,k);
                C((i-1)*2+1+1,(j-1)*2+1,k) = S(i,j,k);
                C((i-1)*2+1+1,(j-1)*2+1+1,k) = S(i,j,k);
            end
        end
    end
end

function M = rot180(S)
    M = rot90(rot90(S));
end

function C = DT_C2q_sigma(DT_C2,C2)
    C = zeros(8,8,12);
    for k = 1:12
       C(:,:,k) = DT_C2(:,:,k) .* (C2(:,:,k)' * (1 - C2(:,:,k)));
    end
end

function B = b2q(DT_C2,C2)
    B = [];
    for q = 1:12
        b = sum(sum((DT_C2(:,:,q).*C2(:,:,q).*(1-C2(:,:,q))))');
        B = [B;b];
    end
end

function DT_k2 = K2pq(S1,DT_C2qs)
    S1R180 = rot180(S1);
    DT_k2 = zeros(5,5,6,12);
    for q = 1:12
        for p = 1:6
            %convn(S1R180(:,:,p),rot180(DT_C2qs(:,:,q)),'valid');
            DT_k2(:,:,p,q) = convn(S1R180(:,:,p),rot180(DT_C2qs(:,:,q)),'valid');
        end
    end
end

function S1 = S1p(DT_C2qs,k2)
    [kl kw kh kn] = size(k2);
    [cl cw ch] = size(DT_C2qs);
    S1 = zeros(12,12,6);
    for p = 1:kh
        S1_ = zeros(12,12,1);
        for q = 1:kn
            S1_ = S1_ + convn(DT_C2qs(:,:,q),k2(:,:,p,q));%paper中k2需要转180度，matlab中需要转180参与计算，这里如果再转180度参与计算，相当于转了360
        end
        S1(:,:,p) = S1_;
    end
end

function C = C1ps(DT_C1p,C1)
    C = zeros(24,24,6);
    for k = 1:6
       C(:,:,k) = DT_C1p(:,:,k) .* (C1(:,:,k)' * (1 - C1(:,:,k)));
    end
end

function R = K11p(I,DT_C1)
    IR180 = rot180(I);
    R = zeros(5,5,6);
    for i = 1:6
        R(:,:,i) = convn(IR180,rot180(DT_C1(:,:,i)),'valid');
    end
end

function B = b1p(C1)
    B = zeros(6,1);
    for i = 1:6
        B(i) = sum(sum(C1(:,:,i))');
    end
end