function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C = 0.3;
sigma = 0.1;



epsilon = 1e-2;
param_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
%param_vec = [0.1 0.3];
%param_vec = [0.1];

n = size(param_vec, 2);
p = zeros(n, n);

%n
%n=1;
for i = 1 : n
    for j = 1 : n
        C = param_vec(i);
        sigma = param_vec(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        p(i, j) = mean(double(pred ~= yval));
    end
end

[a_val, a_index] = max(p, [], 2);
[b_val, b_index] = max(a_val, [], 1);
row_index = b_index;
[temp_val, column_index] = max(p(row_index, :), [], 2);



%a_val
%a_index
%b_val
%b_index
%row_index
%column_index

%C=param_vec(2);
%sigma=param_vec(1);
%C = param_vec(1, row_index);
%sigma = param_vec(1, column_index);

C = param_vec(row_index);
sigma = param_vec(column_index);


p
C
sigma


% =========================================================================

end
