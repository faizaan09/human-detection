load('./hw2data/q3_2_data.mat');
max_epoch = 50;
eta_0 = 1;
eta_1 = 1000;
% data_dims = size(trD);
W = zeros(size(trD,1),length(unique(trLb)));
C = 0.1;
batch_size = 20;
% clf = MultiClassSVM(W,C,eta_0,eta_1);
clf = No_Multiclass_SVM(max_epoch, batch_size, eta_0, eta_1, C);
% [train_loss, val_loss] = clf.train(trD, trLb, valD, valLb, batch_size,max_epoch);
[clf, train_loss, val_loss] = clf.train(trD, trLb, valD, valLb);

% c= 0.1
% eta_0 1
% et1 = 1000