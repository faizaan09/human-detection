% load('./hw2data/q3_2_data.mat');
% max_epoch = 2000;
eta_0 = 1;
eta_1 = 100;
% data_dims = size(trD);
W = zeros(size(trD,1),length(unique(trLb)));
C = 0.1;
[W_final, train_loss, val_loss] = sgd_multiclass_svm(max_epoch,eta_0,eta_1,trD,trLb,W,C,valD,valLb);
%get_accuracy(W_final,trD,trLb)