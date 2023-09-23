format compact; clear; clc; close all;
load mnist5k.mat
d = size(Xtr,2); % dimension of the data, which is 784 in MNIST case

%% Perform gradient descent to find optimal w
w = zeros(d,1); % initial guess for w in R^d

maxIter = 2000; % do 2000 iterations of gradient descent
eta = 0.1; % step size (a.k.a. learning rate) for gradient descent
Acc_train = zeros(maxIter,1); 
Acc_test = zeros(maxIter,1);

for iter = 1:maxIter
    fprintf("iter: %d\n", iter);
    gradf = eval_Gradf(w,Xtr,Ytr); % evaluate the gradient of
    % f(w) using current weight on training data
    
    w = w - eta*gradf;
    % gradient descent iteration; overwrite the old weight vector
    
    Ytr_pred = predict(w,Xtr);
    % Complete this part using the custom function 'predict';
    % find the predicted labels for training images using current weights
    
    Yte_pred = predict(w,Xte);
    % Complete this part using the custom function 'predict';
    % find the predicted labels for test images using current weights
    
    acc_train = eval_ACC(Ytr_pred,Ytr);
    % Complete this part using the custom function 'eval_ACC';
    % evaluate the current prediction accuracy on training set
    
    Acc_train(iter) = acc_train;
    % record the current training accuracy
    
    acc_test = eval_ACC(Yte_pred,Yte);
    % Complete this part using the custom function 'eval_ACC';
    % evaluate the current prediction accuracy on testing set
    
    Acc_test(iter) = acc_test;
    % record the current testing accuracy
end

%% Plot the accuracy curves for training and testing dataset
figure(2);
plot(1:maxIter,Acc_train,1:maxIter,Acc_test,'LineWidth',2);
title('Accuracy Curves During Gradient Descent','FontSize', 16);
xlabel('Iteration','Fontsize',16);
ylabel('Accuracy (%)','Fontsize',16);
legend({'Training accuracy','Testing accuracy'},'Fontsize',16,'Location','best');
