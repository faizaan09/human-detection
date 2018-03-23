load('./hw2data/q3_1_data.mat');
max_epoch = 2000;
eta_0 = 1;
eta_1 = 100;
% data_dims = size(trD);
W = zeros(size(trD,1),length(unique(trLb)));
C = 0.1;
% batch_size = 20;
clf = MultiClassSVM(W,C,eta_0,eta_1);
[clf,trainLoss, valLoss] = clf.train(trD, trLb, valD, valLb, batch_size,max_epoch);
% csvwrite('submission_new.csv',clf.predict(tstD))

% c= 0.1
% eta_0 1
% et1 = 1000

% c, e0,e1,epochs
% 0.1, 1, 1000,50, 80.09
% 0.1, 0.1, 150,30, 80.33, batch_size =10