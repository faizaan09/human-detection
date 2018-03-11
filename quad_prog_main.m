load('./hw2data/q3_1_data.mat');
X = trD;
Y = trLb;
C = 10;

kernelSVM = KernelSVM([],[]);
[kernelSVM,obj_val,alpha] = kernelSVM.train(X,Y,C);
y_pred = kernelSVM.predict(valD);
num_support_vectors = kernelSVM.count_support_vectors(alpha);
[C_m, order] = confusionmat(valLb,y_pred);
acc = kernelSVM.get_accuracy(y_pred,valLb);
acc_train = kernelSVM.get_accuracy(kernelSVM.predict(trD),trLb);

[C_m, order] = confusionmat(valLb,y_pred);