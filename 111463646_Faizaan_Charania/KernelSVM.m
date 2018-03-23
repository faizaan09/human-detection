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
        function [obj,obj_val,alpha] = train(obj,x_train,y_train,C)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
     
            [d,n] = size(x_train);
            H = diag(y_train)*obj.lin_kernel(x_train)*diag(y_train);
            f = -1*ones(1,n);
            A = zeros(1,n);
            A = [];
            b = [];
            beq = 0;
            Aeq = y_train';
            ub = C*ones(n,1);
            lb = zeros(n,1); 
            [alpha,obj_val] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
            obj.W = sum(diag(alpha)*diag(y_train)*x_train')';
            obj.b = mean(y_train((alpha>0 & alpha<C),1) - x_train(:,(alpha>0 & alpha<C))'*obj.W);
%             obj.b = y_train(:,1) - x_train(:,2)'*obj.W
        end
        
       function y_pred = predict(obj,X)
            y_pred = sign(X'*obj.W+obj.b*ones(size(X,2),1));
       end
        
       function num = count_support_vectors(~,alpha)
%             temp = X'*obj.W+obj.b*ones(size(X,2),1);
%             num = numel(temp(temp>=-1&temp<=1));
              num  = numel(alpha(round(alpha,4)>0));
                
       end
       
       function vecs = get_support_vectors(~,alpha)
            vecs = round(alpha,4)>0;
       end
       
        
       function [accuracy] = get_accuracy(obj,y_pred,y_true)
           accuracy = sum((y_pred == y_true))/size(y_true,1)*100;
       end
        
       function kernel_val = lin_kernel(obj,X)
           kernel_val = X'*X; 
       end
        
       function prec = precision(~,y_pred,y_true)
           cm = confusionmat(y_true,y_pred);
%            disp(cm)
           if  sum(cm(:,2)) == 0
            prec = -1;
           end
           prec = cm(2,2)/sum(cm(:,2));
       end
       
       
    end
end

