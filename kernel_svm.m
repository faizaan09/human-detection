% load('./hw2data/q3_1_data.mat');
X = trD;
Y = trLb;
C = 10;

[d,n] = size(X);
H = diag(Y)*lin_kernel(X)*diag(Y);
f = -1*ones(1,n);
A = zeros(1,n);
b = 0;
beq = 0;
Aeq = Y';
ub = C*ones(n,1);
lb = zeros(n,1); 
[alpha,~] = quadprog(H,f,A,b,Aeq,beq,lb,ub);

W = sum(alpha*Y'*X')';
b = sum(Y - X'*W);

y_pred = predict(W,b,valD);
acc = get_accuracy(y_pred,valLb);

function y_pred = predict(W,b,X)
            y_pred = sign(X'*W+b*ones(size(X,2),1));
end

function [accuracy] = get_accuracy(y_pred,y_true)
    accuracy = sum((y_pred == y_true))/size(y_true,1)*100;
end



function kernel_val = lin_kernel(X)
    kernel_val = X'*X; 
end