clc;clear all;
load('NR_ANN_INPUT'); 
load('NR_ANN_OUTPUT');
[~,in] = size(INPUT1); 
[pat,out] = size(OUTPUT1); 
hid = 47;
load('inputweight');
load('outputweight');
a = min(INPUT1(:));
b = max(INPUT1(:)); 
aa = min(OUTPUT1(:));
bb = max(OUTPUT1(:)); 
ra = 0.9; rb = 0.1;
P = (((ra-rb)*(INPUT1 - a)) / (b - a)) + rb;
do = (((ra-rb)*(OUTPUT1 - aa)) / (bb - aa)) + rb;
nn = 0.35; 
lim =0.7*pat+1;
itr = 0;
mse_plot_test = [];
errorr = zeros(1,out);
tic;doo = zeros(1,out);
for u = lim:pat
    oi=P(u,:);
    Netj = oi*V;
    oj=1./(1+exp(-Netj));
    Netk = oj*W;
    ok=1./(1+exp(-Netk));
    error = do(u,:) - ok;
    mse(u) = sum(error.*error)/in;
    mse_sum = sum(mse);
    mse_plot_test = [mse_plot_test mse_sum];
    %Denomalized calculated output
    doo(u-lim+1,:) = ((ok -rb).*(bb-aa)./(ra-rb))+aa;
    %Actual error
    errorr(u-lim+1,:) = OUTPUT1(u,:) - doo(u-lim+1,:); 
end
toc;
%Plot for actual values of all training patterns
plot(errorr);
mse_plot_test;
hold on;
%Mean square error plot
plot(mse_plot_test, 'LineWidth',5)
hold off;
h = input('Enter pattern number(1-30) to observe: ');
calculated = doo(h,:)';
actual  = OUTPUT1(h+70,:)';
errr = calculated - actual;
percentage_err = (errr./actual)*100;
observation = [calculated actual errr percentage_err];
disp('-----------------------------------------')
disp(' cal(o/p)  target(o/p)   error   %error');
disp('-----------------------------------------')
disp(observation);