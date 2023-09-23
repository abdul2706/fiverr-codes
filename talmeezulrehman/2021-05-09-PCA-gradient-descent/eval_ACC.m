function acc = eval_ACC(Y_pred, Y_true)
% This function evaluates the prediction accuracy for a given 
% pair of predicted labels and true labels
%
% inputs: Y_pred: predicted label vector for a set of images 
%         Y_true: the corresponding true label vector
% ouput:  acc: evaluated prediction accuracy in percentage

n = length(Y_pred); % total numbner of images
acc = 100 * sum(Y_pred == Y_true) / n; % Complete this part

end
