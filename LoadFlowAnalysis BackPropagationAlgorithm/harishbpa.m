clc;clear all;
load('NR_ANN_INPUT'); 
load('NR_ANN_OUTPUT');                         
[~,in] = size(INPUT1); 
[pat,out] = size(OUTPUT1); 
hid = 47;
V=-1+2*rand(in, hid); 
W=-1+2*rand(hid,out);
a = min(INPUT1(:));
b = max(INPUT1(:));
aa = min(OUTPUT1(:)); 
bb = max(OUTPUT1(:)); 
ra = 0.9; rb = 0.1;
P = (((ra-rb) * (INPUT1 - a)) / (b - a)) + rb;                                    
do = (((ra-rb) * (OUTPUT1 - aa)) / (bb - aa)) + rb;                          
nn = 0.25; lim =0.7*pat; 
itr = 0; tol = 1e-4; alp = 0.01;
itr = 0; e=1; eee=[];
dW = zeros(hid,out);
dV = zeros(in,hid);
while e>tol
    itr = itr + 1;
    for s = 1: lim
        %input data coloum vector
        oi=P(s,:);
        %output of first layer "i" is same as input
%         for j = 1:hid
%             for i=1:in
%                 f=V(i,j)*oi(i);
%                 Netj(j) = Netj(j)+f;
%             end
%         end
        Netj = oi*V;
        %finding output of hidden layer "j"
%         for j=1:hid
%             oj(j)=1/(1+exp(-Netj(j)));
%         end
        oj = 1./(1+exp(-Netj));
        %output of hidden layer
%         for k = 1:out
%             for j=1:hid
%                 q=W(j,k)*oj(j);
%                 Netk(k) = Netk(k)+q;
%             end
%         end
        Netk =oj*W;
        %finding output of output layer "k"
%         for k=1:out
%             ok(k)=1/(1+exp(-Netk(k)));
%         end
        ok = 1./(1+exp(-Netk));
        %output of output layer
        %Error calculation
%         for k=1:out
%             error(k)=(do(s,k)-ok(k));
%         end
        error = do(s,:) - ok;
        me(s)=sum(error.*error)/out;
        %disp('Back propagation');
%         for k=1:out
%             delk(k)=(do(s,k)-ok(k))*ok(k)*(1-ok(k));
%         end
        delk=error.*ok.*(1-ok);
%         for k = 1:out
%             for j=1:hid
%                 dW(j,k)=alp*dW(j,k)+nn*delk(k)*oj(j);
%             end
%         end
        dW=alp*dW+nn.*(oj'*delk);
        W = W+dW;
%         for j=1:hid
%             for k=1:out
%                 summ=delk(k)*W(j,k);
%                 x(j) = x(j) + summ;
%             end
%             delj(j)=x(j)*oj(j)*(1-oj(j));
%         end
%         for i=1:in
%             for j=1:hid
%                 dV(i,j)=alp*dV(i,j)+nn*delj(j)*oi(i);
%             end
%         end
        su=delk*W';
        delj=su.*oj.*(1-oj);
        dV=alp*dV+nn.*oi'*delj;
        V=V+dV; 
    end
    e=sum(me);
    eee=[eee e];
end
plot(eee,'r')