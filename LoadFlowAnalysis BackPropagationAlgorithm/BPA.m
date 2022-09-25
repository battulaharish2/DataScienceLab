clc; clear all;
load('NR_ANN_INPUT'); load('NR_ANN_OUTPUT');
[~,in] = size(INPUT1); [pat,out] = size(OUTPUT1); hid = 47;
%Weight matrices
V=-1+2*rand(in, hid); W=-1+2*rand(hid,out);
%patterns generation
a = min(INPUT1(:));b = max(INPUT1(:));
aa = min(OUTPUT1(:));bb = max(OUTPUT1(:));
ra = 0.9;rb = 0.1;
P = (((ra-rb) * (INPUT1 - a)) / (b - a)) + rb;
do = (((ra-rb) * (OUTPUT1 - aa)) / (bb - aa)) + rb;
nn = 0.25; lim =0.7*pat;
tic;
for s = 1: lim
    %input data coloum vector
    X1=P(s,:);
    Netj=zeros(1,hid); Netk=zeros(1,out); x=zeros(1,hid);
    oj = zeros(1,hid); ok = zeros(1,out); delk = zeros(1,out);
    delj = zeros(1,hid); dW = zeros(hid,out); dV = zeros(in,hid);
    errror = zeros(1,in); itr = 0; ek = 1; e=1;
    while e>1e-3
        %output of first layer "i" is same as input
        oi=X1; e=0;
        for j = 1:hid
            for i=1:in
                f=V(i,j)*oi(i);
                Netj(j) = Netj(j)+f;
            end
        end
        Netj;
        %finding output of hidden layer "j"
        for j=1:hid
            oj(j)=1/(1+exp(-Netj(j)));
        end
        oj;
        %output of hidden layer
        for k = 1:out
            for j=1:hid
                q=W(j,k)*oj(j);
                Netk(k) = Netk(k)+q;
            end
        end
        Netk;
        %finding output of output layer "k"
        for k=1:out
            ok(k)=1/(1+exp(-Netk(k)));
        end
        ok;
        %output of output layer
        %Error calculation
        for k=1:out
            error(k)=((do(s,k)-ok(k))^2)./out;
            e = sum(error);
        end
        e;
        if e ==ek
            break;
        end
        %disp('Back propagation');
        for k=1:out
            delk(k)=(do(s,k)-ok(k))*ok(k)*(1-ok(k));
        end
        for k = 1:out
            for j=1:hid
                dW(j,k)=nn*delk(k)*oj(j);
            end
        end
        W = W+dW;
        for j=1:hid
            for k=1:out
                su=delk(k)*W(j,k);
                x(j) = x(j) + su;
            end
            delj(j)=x(j)*oj(j)*(1-oj(j));
        end
        for i=1:in
            for j=1:hid
                dV(i,j)=nn*delj(j)*oi(i);
            end
        end
        V=V+dV;
        ek = e;
        itr = itr + 1;
    end
    iter(s) = itr;
    %plot(error)
    errtn(s) = e;
end
toc;
tic;
for u = lim : pat
    oi = P(u,:);
    Netj=zeros(1,hid);
    Netk=zeros(1,out);
    x=zeros(1,hid);
    oj = zeros(1,hid);
    ok = zeros(1,out);
    delk = zeros(1,out);
    delj = zeros(1,hid);
    for j = 1:hid
        for i=1:in
            f=V(i,j)*oi(i);
            Netj(j) = Netj(j)+f;
        end
    end
    %finding output of hidden layer "j"
    for j=1:hid
        oj(j)=1/(1+exp(-Netj(j)));
    end
    %output of hidden layer
    for k = 1:out
        for j=1:hid
            q=W(j,k)*oj(j);
            Netk(k) = Netk(k)+q;
        end
    end
    %finding output of output layer "k"
    for k=1:out
        ok(k)=1/(1+exp(-Netk(k)));
    end
    for k=1:out
        error(k)=((do(u,k)-ok(k))^2)./out;
        e = max(error);
    end
        errtn(u) = e;
end
plot(errtn,'r');
toc;