clc;clear all;
load('NR_ANN_INPUT'); load('NR_ANN_OUTPUT');                    % load input and target data
[~,in] = size(INPUT1); [pat,out] = size(OUTPUT1); hid = 47;     % Found input, hidden, output nodes
V=-0.8+1.6*rand(in, hid); W=-0.8+1.6*rand(hid,out);             % Ininitialized weight matrices
a = min(INPUT1(:));b = max(INPUT1(:)); 
aa = min(OUTPUT1(:));bb = max(OUTPUT1(:));ra = 0.9;rb = 0.1;
P = (((ra-rb) * (INPUT1 - a)) / (b - a)) + rb;                  % Normalized input data
do = (((ra-rb) * (OUTPUT1 - aa)) / (bb - aa)) + rb;             % Normalized target data
nn = 0.35; lim =0.7*pat; itr = 0; tol = 1e-4;                   % Initialized learning rate and number of samples for tarining and tolerance of error.
mse_plot_train = []; sum_mse = 1;alp = 0.01;
dW = zeros(hid,out);dV = zeros(in,hid);
tic;
while sum_mse>tol
    itr = itr +1;
    for s = 1: lim
        oi=P(s,:);                           
        % assaigning pattern
        Netj = oi*V;
        oj=1./(1+exp(-Netj));                
        % Calculating output of hidden layer
        Netk = oj*W;
        ok=1./(1+exp(-Netk));                
        % Calculating output of output layer
        error = do(s,:) - ok;                
        % Calculating error
        mse(s) = sum(error.*error)/out;      
        % Calculating mean square error
        delk=error.*ok.*(1-ok);
        dW=alp*dW+nn.*(oj'*delk);
        W = W+dW;
        % Updating output weight matrix
        su=delk*W';
        delj=su.*oj.*(1-oj);
        dV=alp*dV+nn.*oi'*delj;
        V=V+dV;                              
        % Updating input weight matrix
    end
    sum_mse = sum(mse);
    mse_plot_train = [mse_plot_train sum_mse]; 
    %storing sum of mse for every iteration
end
toc;
plot([1:itr],mse_plot_train,'LineWidth',3);
title('mean square error plot convergence using BPA');
fprintf('No.of itterations: %d\n',itr);
fprintf('Tolerance considered: %d\n',tol);
fprintf('error convereged at: %d\n', sum_mse);