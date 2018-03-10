load('./hw2data/q3_1_data.mat');
X = trD;
% clear trD
Y = trLb;
% clear trLb
C = 10;

alpha = train_ksvm_dual(trD, trLb, 0.1);

[d,n] = size(X);
X = X';
Y =Y';


H = diag(Y)*lin_kernel(X)*diag(Y);
f = -1*ones(1,n);
A = zeros(1,n);
b = 0;
beq = 0;
Aeq = Y;
ub = C*ones(n,1);
lb = zeros(n,1); 
[alpha,~] = quadprog(H,f,A,b,Aeq,beq,lb,ub);


W = sum(X*Y*alpha',2);
b = sum(Y - X'*W);

y_pred = predict(W,b,valD);
acc = get_accuracy(y_pred,valLb);

function y_pred = predict(W,b,X)
            y_pred = X'*W+b*ones(size(X,2),1);
end

function [accuracy] = get_accuracy(y_pred,y_true)
    accuracy = sum((y_pred == y_true))/size(y_true,1)*100;
end



function kernel_val = lin_kernel(X)
    kernel_val = X*X'; % dimensions reversed
end


function [alpha] = train_ksvm_dual(X, y, C)
   
    X = transpose(X);
    y = transpose(y);
    
    m = size(X,1);
    gram= zeros(m);
    gram = X*transpose(X);
 
    H = diag(y)*gram*diag(y);
    f = -1*ones(1,m);
    A = zeros(1,m);
    b = 0;
    Aeq = y;
    beq = 0;
    lb = zeros(m,1);
    ub = C*ones(m,1);
    [alpha,fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    
end