load('./hw2data/q3_1_data.mat');
X = trD;
Y = trLb;
C = 10;

kernelSVM = KernelSVM([],[]);
kernelSVM = kernelSVM.train(X,Y,C);
y_pred = kernelSVM.predict(valD);
acc = kernelSVM.get_accuracy(y_pred,valLb);
acc_train = kernelSVM.get_accuracy(kernelSVM.predict(trD),trLb);