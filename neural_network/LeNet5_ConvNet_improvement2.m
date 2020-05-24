%k1 = unifrnd(-sqrt(6/((1+6)*5.^2)),sqrt(6/((1+6)*5.^2)),5,5,6);
%k32 = unifrnd(-sqrt(6/((6+32)*5.^2)),sqrt(6/((6+32)*5.^2)),5,5,6,32);
%W32 = unifrnd(-sqrt(6/(10+512)),sqrt(6/(10+512)),10,512);
%增加了第二个卷积层的filter数，从12增加到了32，在a=0。001的情况下，可以获得0。9152的准确率
%将激活函数换成了relu，avg poolinng换成了max pooling，第二个卷积层换成了32个filter。
%当第二个卷积层提升到36个的时候，会发现并不能有效提高正确率，说明需要改进第一层。
%这个版本中对第一层的filter数量进行增加。
%
%improvement2 params:
%k12 = unifrnd(-sqrt(12/((1+12)*5.^2)),sqrt(6/((1+12)*5.^2)),5,5,12);
%k1248 = unifrnd(-sqrt(12/((12+48)*5.^2)),sqrt(12/((12+48)*5.^2)),5,5,12,48);
%正确率可以提高到0。94
function [k1,B1,k2,B2,W,B3] = LeNet5_ConvNet_improvement2(mnist,k1,k2,W,a)
    TRAINING = mnist.training;
    TR_LABELS = TRAINING.labels;
    TEST = mnist.test;
    T_LABELS = TEST.labels;
    k1h = size(k1,3);
    B1 = zeros(1,1,k1h);
    n = size(k2,4);
    B2 = zeros(1,1,n);
    B3 = zeros(10,1,1);
    
    for i = 1:60000
        %feedforward
        R1 = conv_C1(TRAINING.images(:,:,i),k1);
        C1 = Relu(R1 + B1);
        S1 = max_pooling(C1);
        R2 = conv_C2(S1,k2);
        C2 = Relu(R2 + B2);
        S2 = max_pooling(C2);
        f = vctz_concat(S2);
        y_hat = logsig(W*f + B3);
        y = label_vctz(TR_LABELS(i));

        %Backpropagation
        DT_Y = (y_hat - y) .* (y_hat' * (1 - y_hat));
        DT_W = DT_Y*f';
        DT_B = DT_Y;
        DT_f = W' * DT_Y;
        DT_S2 = reverse_vctz_concat(DT_f);%!
        DT_C2 = reverse_avg_pooling4(DT_S2);
        DT_C2qs = C2qRelu(DT_C2,C2);
        DT_K2pq = K2pq(S1,DT_C2qs);
        DT_B2q = b2q(DT_C2,C2);
        DT_S1_p = S1p(DT_C2qs,k2);
        DT_C1p = reverse_avg_pooling4(DT_S1_p);
        DT_C1ps = C1pRelu(DT_C1p,C1);
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
        C1 = Relu(R1 + B1);
        S1 = max_pooling(C1);
        R2 = conv_C2(S1,k2);
        C2 = Relu(R2 + B2);
        S2 = max_pooling(C2);
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

%替换avg_pooling，max pooling比avg pooling效果要好。
function P = max_pooling(C)
    [cl cw ch] = size(C);
    
    P = zeros(cl/2,cw/2,ch);
    for k = 1:ch
        for i = 1:(cl/2)
            for j = 1:(cw/2)
                c1 = C(2*i-1, 2*j-1, k);
                c2 = C(2*i-1, 2*j, k);
                c3 = C(2*i, 2*j-1, k);
                c4 = C(2*i, 2*j, k);
                a = max([c1;c2;c3;c4]);
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
    n = size(f,1)/16;
    S = zeros(4,4,n);
    for i = 0:n-1
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

%当使用maxpooling的时候，这里取1/4效果会好一些。可以替代不取1/4的方法。
function C = reverse_avg_pooling4(S)%！取1/4
    [l w h] = size(S);
    C = zeros(l*2,w*2,h);
    for k = 1:h
        for i = 1:l
            for j = 1:w
                C((i-1)*2+1,(j-1)*2+1,k) = S(i,j,k)/4;
                C((i-1)*2+1,(j-1)*2+1+1,k) = S(i,j,k)/4;
                C((i-1)*2+1+1,(j-1)*2+1,k) = S(i,j,k)/4;
                C((i-1)*2+1+1,(j-1)*2+1+1,k) = S(i,j,k)/4;
            end
        end
    end
end

function M = rot180(S)
    M = rot90(rot90(S));
end

function C = C2q_sigma(DT_C2,C2)
    [cl cw ch] = size(C2);
    C = zeros(8,8,ch);
    for k = 1:ch
       C(:,:,k) = DT_C2(:,:,k) .* (C2(:,:,k)' * (1 - C2(:,:,k)));
    end
end

function C = C2qRelu(DT_C2,C2)
    [cl cw ch] = size(C2);
    C = zeros(8,8,ch);
    for k = 1:ch
       C(:,:,k) = DT_C2(:,:,k) .* Relu_diff(C2(:,:,k));
    end
end

function D = Relu_diff(M)
    [l w h n] = size(M);
    D = zeros(l,w,h,n);
    lg = M > 0;
    D(lg) = 1;
end

function M = Relu(M)
    lg = (M <= 0);
    M(lg) = 0;
end

function B = b2q(DT_C2,C2)
    h = size(C2,3);
    B = [];
    for q = 1:h
        b = sum(sum((DT_C2(:,:,q).*C2(:,:,q).*(1-C2(:,:,q))))');
        B = [B;b];
    end
end

%这里的卷积操作很关键
function DT_k2 = K2pq(S1,DT_C2qs)
    ph = size(S1,3);
    ch = size(DT_C2qs,3);
    S1R180 = rot180(S1);
    DT_k2 = zeros(5,5,6,16);
    for q = 1:ch
        for p = 1:ph
            %convn(S1R180(:,:,p),rot180(DT_C2qs(:,:,q)),'valid');
            DT_k2(:,:,p,q) = convn(S1R180(:,:,p),rot180(DT_C2qs(:,:,q)),'valid');
        end
    end
end

%这个方法很关键，是卷积层里反向传播的关键！
function S1 = S1p(DT_C2qs,k2)
    [kl kw kh kn] = size(k2);
    [cl cw ch] = size(DT_C2qs);
    S1 = zeros(kl+cl-1,kw+cw-1,kh);
    for p = 1:kh
        S1_ = zeros(kl+cl-1,kw+cw-1,1);%!反向求出被卷积的矩阵大小。
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

function C = C1pRelu(DT_C1p,C1)
    [l w h] = size(DT_C1p);
    C = zeros(l,w,h);
    for k = 1:h
       C(:,:,k) = DT_C1p(:,:,k) .* Relu_diff(C1(:,:,k));
    end
end

function R = K11p(I,DT_C1)
    [il iw ih] = size(I);
    [cl cw ch] = size(DT_C1);
    IR180 = rot180(I);
    R = zeros(il-cl+1,iw-cw+1,ch);
    for i = 1:ch
        R(:,:,i) = convn(IR180,rot180(DT_C1(:,:,i)),'valid');
    end
end

function B = b1p(C1)
    h = size(C1,3);
    B = zeros(h,1);
    for i = 1:h
        B(i) = sum(sum(C1(:,:,i))');
    end
end