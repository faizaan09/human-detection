function [W,train_loss,val_loss] = sgd_multiclass_svm(max_epoch,eta_0,eta_1,X,Y,W,C,x_val,y_val);
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    train_loss = zeros(max_epoch,1);
    val_loss = zeros(max_epoch,1);
    classes = unique(Y);
    for epoch = 1:max_epoch

        eta = eta_0/(eta_1 + epoch);
        count = 0;
        gradient_L = 0;
        for i = randperm(size(X,2))
            count = count +1;
            temp = W' * X(:,i);
            temp(Y(i,1) == classes) = [];
            [~, i_cap] = max(temp);
            if i_cap >= i
                i_cap = i_cap + 1;
            end
            gradient_L = gradient_L + get_subgradient();
            if mod(count,793) == 0
                W = W - eta*gradient_L;
                gradient_L = 0;
            end
        end
        train_loss(epoch) = get_error(W,X,Y);
        val_loss(epoch) = get_error(W,x_val,y_val);
        if mod(epoch,100) == 0
            disp('Loss and accuracy')
            disp([train_loss(epoch),get_accuracy(W,X,Y)])
            disp([val_loss(epoch),get_accuracy(W,x_val,y_val)])
        end
    end
    
    function [gradient] = get_subgradient () %i_cap,C,X,Y,W)
        gradient = zeros(size(W));
        for ind = 1:size(gradient,2)
            if (W(:,i_cap)'*X(:,i) +1 > W(:,Y(i,1) == classes)'*X(:,i))
                if ind == find(Y(i,1) == classes)
                    gradient(:,ind) = - C* X(:,i);
                elseif ind == find(Y(i_cap,1) == classes)  
                    gradient(:,ind) = C *X(:,i);
                end
            end
            gradient(:,ind) =gradient(:,ind) + W(:,ind)/size(X,2);
        end
        
    end
end

function [error] = get_error(W,X,Y)
    error = 0;
    classes = unique(Y);
    for i = 1:size(X,2)
        temp = W' * X(:,i);
        temp(Y(i,1) == classes) = [];
        [~, i_cap] = max(temp);
        if i_cap >= i
            i_cap = i_cap + 1;
        end    
        error = error + max(0,W(:,i_cap)' * X(:,i) - W(:,Y(i,1) == classes)' * X(:,i) + 1);
    end

end