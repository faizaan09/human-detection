function [accuracy] = get_accuracy(W,X,Y)

    [~,y_pred] = max(W'*X);
    accuracy = sum((y_pred' == Y))/size(Y,1)*100;
end