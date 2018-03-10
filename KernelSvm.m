classdef KernelSVM
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        W
        b
    end
    
    methods
 
        function obj = KernelSVM(W,b)
            obj.W = W;
            obj.b = b;
        end
        function obj = train(obj,x_train,y_train,C)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
     
            [d,n] = size(x_train);
            H = diag(y_train)*obj.lin_kernel(x_train)*diag(y_train);
            f = -1*ones(1,n);
            A = zeros(1,n);
            b = 0;
            beq = 0;
            Aeq = y_train';
            ub = C*ones(n,1);
            lb = zeros(n,1); 
            [alpha,~] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
            obj.W = sum(alpha*y_train'*x_train')';
            obj.b = mean(y_train(1,1) - x_train(:,1)'*obj.W);

        end
        
       function y_pred = predict(obj,X)
            y_pred = sign(X'*obj.W+obj.b*ones(size(X,2),1));
        end
        
        function [accuracy] = get_accuracy(obj,y_pred,y_true)
            accuracy = sum((y_pred == y_true))/size(y_true,1)*100;
        end
        
        function kernel_val = lin_kernel(obj,X)
            kernel_val = X'*X; 
        end

    end
end

