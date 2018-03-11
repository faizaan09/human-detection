classdef MultiClassSVM
    properties
        W
        C
        eta_0
        eta_1
    end
    
    
    methods
        function obj = MultiClassSVM(W,C,eta_0,eta_1)
            obj.W = W;
            obj.C = C;
            obj.eta_0 = eta_0;
            obj.eta_1 = eta_1;
        end
        
        function [train_loss,val_loss] = train(obj, x_train, y_train, x_val, y_val, batch_size,max_epoch)
        %UNTITLED3 Summary of this function goes here
        %   Detailed explanation goes here
%             obj.W = W;
            train_loss = zeros(max_epoch,1);
            val_loss = zeros(max_epoch,1);
            classes = unique(y_train);
            for epoch = 1:max_epoch

                eta = obj.eta_0/(obj.eta_1 + epoch);
                count = 0;
                gradient_L = 0;
                for i = randperm(size(x_train,2))
                    count = count +1;
                    temp = obj.W' * x_train(:,i);
                    temp(y_train(i,1) == classes) = -Inf;
                    [~, i_cap] = max(temp);
                    gradient_L = gradient_L + obj.get_subgradient(x_train,y_train,i,i_cap,classes);
                    if mod(count,batch_size) == 0
                        obj.W = obj.W - eta*gradient_L;
                        gradient_L = 0;
                    end
                end
                train_loss(epoch) = calculate_loss(obj,x_train,y_train);
                val_loss(epoch) = calculate_loss(obj,x_val,y_val);
                if mod(epoch,batch_size) == 0 || epoch ==1
                    fprintf('Epoch %d\n',epoch)
                    fprintf('%f %f\n',train_loss(epoch),obj.get_accuracy(obj.predict(x_train),y_train))
                    fprintf('%f %f\n',val_loss(epoch),obj.get_accuracy(obj.predict(x_val),y_val))
                end
            end
    
        end
        
        function [gradient] = get_subgradient (obj,X,Y,i,i_cap,classes) %i_cap,C,X,Y,W)
            gradient = zeros(size(obj.W));
            yi = Y(i,1) == classes;
            xi = X(:,i);
            for ind = 1:size(gradient,2)
                if (obj.W(:,i_cap)'*xi +1 > obj.W(:,yi)'*x)
                    if ind == find(Y(i,1) == classes)
                        gradient(:,ind) = - obj.C* X(:,i);
                    elseif ind == i_cap  
                        gradient(:,ind) = obj.C *X(:,i);
                    end
                end
                gradient(:,ind) =gradient(:,ind) + obj.W(:,ind)/size(X,2);
            end
        end
        
        function [error] = calculate_loss(obj,X,Y)
            error = 0;
            classes = unique(Y);
            for i = 1:size(X,2)
                temp = obj.W' * X(:,i);
                temp(Y(i,1) == classes) = -Inf;
                [~, i_cap] = max(temp);
                error = error + max(0,obj.W(:,i_cap)' * X(:,i) - obj.W(:,Y(i,1) == classes)' * X(:,i) + 1);
            end

        end
        
        function y_pred = predict(obj,X)
            [~,y_pred] = max(obj.W'*X);
        end
        
        function [accuracy] = get_accuracy(obj,y_pred,y_true)
            accuracy = sum((y_pred' == y_true))/size(y_true,1)*100;
        end
    end
end