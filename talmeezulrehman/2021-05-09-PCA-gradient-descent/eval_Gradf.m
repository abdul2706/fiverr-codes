function gradf = eval_Gradf(w,X,Y)
% This function evaluate the gradient with respect to weights
% of loss function f(w) over the dataset {X, Y}
%
% inputs: w: weight vector in R^d for logistic regression model
%         X: n-by-d image matrix; each row is an image in R^d
%         Y: true label vector for images in X
% output: gradf: gradient vector in R^d of f(w) w.r.t. w

[n,d] = size(X); % get the number of image samples and number of weights

gradf = zeros(d,1); % Complete this part
for i=1:n
    numerator = -Y(i) .* exp(-Y(i) * (X(i, :) * w)) .* X(i, :)';
    denominator = 1 + exp(-Y(i) * (X(i, :) * w));
    gradf = gradf + (numerator ./ denominator);
end
gradf = gradf / n;

end
